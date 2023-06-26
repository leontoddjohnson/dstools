from ..ds_util import Slicer
import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import MinMaxScaler


class PeriodicRank(object):
    def __init__(self, segment_features, target_value, similarity_features=None):
        '''

        Parameters
        ----------
        segment_features
        target_value : str
            The values in this column must reflect:
                non-NA = the target happened with some value
                NA = the target did not happen
        '''
        # TODO: Change the name 'segment' if you need to (check to make sure it's not a segment_feature)
        self.segment_features = segment_features
        self.similarity_features = similarity_features
        self.target_value = target_value
        self.df_segments = None
        self.df_sched_expected = None
        self.defaults = None

    def _get_segments(self, df):
        '''
        Reusable to define segment indices to combinations of segment_feature values.
        '''
        segments = Slicer(self.segment_features,
                          value=self.target_value,
                          agg_list=[(f'{self.target_value}_segment_mean', 'mean')])

        df_segments = segments.get_segments(df)

        return df_segments

    def _get_hist_segments(self, df):
        '''
        Given historical data to start with, determine the initial segments based on segment features provided.

        Parameters
        ----------
        df : pandas.DataFrame
            Initial data set with segment features
        '''
        df_segments = self._get_segments(df)

        # Save the date that these segments are considered "new"
        new_as_of = pd.to_datetime(pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
        df_segments['new_as_of'] = new_as_of

        self.df_segments = df_segments

    def _add_segments(self, df_update):
        '''
        Given new data, determine segments (using same segment features), and add new ones to the ongoing set of
        segments. Update means and counts within the segment.
        '''
        n_orig = self.df_segments.shape[0]
        # Get the segment indices for the updating segments. Some have likely been seen before
        df_update_segments = self._get_segments(df_update)

        # Merge historical and update segments to find which ones are new
        keepcols = self.segment_features + [f'{self.target_value}_segment_mean', 'num_in_segment']
        df_segments_joined = self.df_segments.merge(df_update_segments[keepcols],
                                                    on=self.segment_features,
                                                    how='outer',
                                                    suffixes=('_hist', '_update'))

        # Update means
        df_segments_joined[f'{self.target_value}_segment_mean'] = df_segments_joined[
                                                                    [f'{self.target_value}_segment_mean_hist',
                                                                     f'{self.target_value}_segment_mean_update']] \
                                                                    .mean(axis=1)

        # Update counts within the segment
        df_segments_joined['num_in_segment'] = df_segments_joined[['num_in_segment_hist',
                                                                   'num_in_segment_update']].sum(axis=1)

        # Drop unnecessary columns
        df_segments_joined.drop(columns=[f'{self.target_value}_segment_mean_update',
                                         f'{self.target_value}_segment_mean_hist',
                                         'num_in_segment_update',
                                         'num_in_segment_hist'],
                                inplace=True)

        # For the new segments (where 'new_as_of' is null), update this date with the current date
        new_as_of = pd.to_datetime(pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
        df_segments_joined['new_as_of'].fillna(new_as_of, inplace=True)

        # Sort segments (contiguous from 0 to n_orig) and fill the missing with new indices (dataframe index)
        df_segments_joined.sort_values('segment', ignore_index=True, inplace=True, na_position='last')
        nullmask = df_segments_joined['segment'].isnull()
        df_segments_joined.loc[nullmask, 'segment'] = df_segments_joined.loc[nullmask].index
        df_segments_joined.loc[:, 'segment'] = df_segments_joined['segment'].astype(int)

        self.df_segments = df_segments_joined

        n_new = self.df_segments.shape[0]
        print(f"Added {n_new - n_orig} segments for a current total of {n_new} segments.")

    def _get_segment_t_mean(self, df_, time_period):
        '''

        '''
        df_segment_t_mean = pd.pivot_table(df_[['segment', time_period, self.target_value]],
                                           index='segment',
                                           values=self.target_value,
                                           columns=time_period,
                                           aggfunc=np.nanmean,
                                           dropna=False)

        return df_segment_t_mean

    def _get_segment_means(self, df_, best_target_predictor):
        '''
        For each segment, calculate the average target value for that segment. If the segment did not have sufficient
        data for the target to take on a value, use the average target value among entities within that segments
        best_target_predictor.

        E.g., I don't know the average payment for Female Detectives, but I think that "detective-ness" is the best
        predictor for payment, and I know the average payment among detectives, so I'll use that.

        '''
        # Get the segments, and the segment feature we'll use to fill "similar" segments
        target_mean = self.target_value + '_segment_mean'
        segment_means = self.df_segments[['segment', best_target_predictor, target_mean]].copy()

        # Given data, calculate the average target for each of the best "target-predicting" segment feature, then merge
        filler = df_.groupby(best_target_predictor)[self.target_value].mean()
        segment_means = segment_means.merge(filler, on=best_target_predictor, how='left')

        # Fill the averages for segments with no target values with the average for our predictor factor
        segment_means.loc[:, target_mean] = segment_means[target_mean].fillna(segment_means[self.target_value])
        segment_means.drop(columns=[best_target_predictor, self.target_value], inplace=True)

        segment_means = pd.Series(data=segment_means[target_mean],
                                  index=segment_means['segment'])

        return segment_means

    def _get_prior(self, df_, return_all=False):
        '''
        Prior probability is the probability that an entity (within any segment) will take an action
        at some point in the observable schedule.
        '''
        df_notnull_in_segment = df_.groupby('segment')[self.target_value] \
                                   .agg([('notnull_in_segment', 'count')]).reset_index()

        df_all_in_segment = df_['segment'].value_counts(dropna=False)
        df_all_in_segment = pd.DataFrame({'segment': df_all_in_segment.index,
                                          'all_in_segment': df_all_in_segment})

        df_prior = pd.merge(df_notnull_in_segment,
                            df_all_in_segment,
                            on='segment')

        df_prior['p'] = df_prior['notnull_in_segment'] / df_prior['all_in_segment']

        if return_all:
            return df_prior
        else:
            return df_prior.set_index('segment')['p']

    @staticmethod
    def _get_evidence(df_, time_period):
        df_['ones'] = 1

        # Number of observances for x segment at t time
        df_counts = pd.pivot_table(df_[['segment', time_period, 'ones']],
                                   index='segment',
                                   values='ones',
                                   columns=time_period,
                                   aggfunc=np.nansum).fillna(0)

        # Number of observances (all) for a segment
        df_sums = df_counts.sum(axis=1)

        df_evidence = df_counts.values / df_sums.values.reshape(-1, 1)

        df_evidence = pd.DataFrame(data=df_evidence,
                                   index=df_counts.index,
                                   columns=df_counts.columns)

        return df_evidence

    def _get_posterior(self, df_, time_period):
        df_count_pos = pd.pivot_table(df_[['segment', time_period, self.target_value]],
                                      index='segment',
                                      values=self.target_value,
                                      columns=time_period,
                                      aggfunc='count',
                                      dropna=True).fillna(0)

        df_sums = df_count_pos.sum(axis=1)

        df_posterior = df_count_pos.values / df_sums.values.reshape(-1, 1)

        # There will be rows of NAN values if that segment has not taken the target action
        df_posterior = pd.DataFrame(data=df_posterior,
                                    index=df_count_pos.index,
                                    columns=df_count_pos.columns)

        return df_posterior

    def _get_bayes_prob(self, df_, time_period):
        '''
        For a given segment, we want to know what is the probability that they will take the target action
        given that we are interacting (or observing) them at time period t. So, *within a segment*, we have:

            A = An entity will take the target action  (prior)
            B = We interact with this entity at time period t  (evidence)

        Bayes Theorem states:
                P(A) / P(B) = P(A|B) / P(B|A)
            --> P(A|B) = P(B|A) * (P(A) / P(B))

        We calculate (for each segment):
            P(A) = (# not-null target values for that segment) / (# all rows in that segment)
            P(B) = (# all rows at time period t for that segment) / (# all rows in that segment)

            P(B|A) = P(B and A) / P(A)  (posterior)

                         # not-null target values for that segment at time t
                   =  ------------------------------------------------------------
                              # not-null target values for that segment

        With frequentist (non-updating/learning) approach, this would equate to:
                          # not-null target values for that segment at time t
            P(A|B) =  ------------------------------------------------------------   ,
                              # all rows at time period t for that segment

            which makes sense
        '''
        df_prior = self._get_prior(df_)
        df_evidence = self._get_evidence(df_, time_period)
        df_posterior = self._get_posterior(df_, time_period)

        df_prior_evidence = df_prior.values.reshape(-1, 1) / df_evidence.values

        df_prior_evidence = pd.DataFrame(data=df_prior_evidence,
                                         index=df_evidence.index,
                                         columns=df_evidence.columns)

        df_probs = df_posterior * df_prior_evidence

        return df_probs

    @staticmethod
    def _get_update_sample(df_hist,
                           event_time,
                           n_lookback_days=28 * 4,
                           max_sample_frac=0.75,
                           scale_min=1,
                           scale_max=10,
                           exp_lambda=0.5,
                           today=None):

        df_hist.loc[:, event_time] = pd.to_datetime(df_hist[event_time])

        if today is None:
            today = df_hist[event_time].max()

        # We base the sample size on the amount of data that we'd expect in the last `lookback_days` days
        sample_size = (df_hist[event_time] >= (today - pd.Timedelta(n_lookback_days, unit='days'))).sum()

        # Assume that "new information" is defined as what is new as of the last *day*
        df_hist.loc[:, event_time] = df_hist[event_time].dt.strftime("%Y%m%d").astype(int)
        most_recent_target = df_hist[event_time].max()

        # The sample should include *all* data from the most recent day
        df_hist_sample_1 = df_hist[df_hist[event_time] == most_recent_target].copy().reset_index(drop=True)

        # We sample from the rest of the data, with higher probability (exponentially) for more recent samples
        df_hist_sample_2 = df_hist[df_hist[event_time] < most_recent_target].copy().reset_index(drop=True)
        sample_size = min(sample_size, int(df_hist_sample_2.shape[0] * max_sample_frac))

        # We calculate weights using an exponential distribution over scaled data
        # With default values, sample starts to trickle down after 6 months
        expon = stats.expon
        sample_weights = df_hist_sample_2[event_time].values.reshape(-1, 1)
        sample_weights = MinMaxScaler((scale_min, scale_max)).fit_transform(sample_weights)
        sample_weights = 1 / expon.pdf(sample_weights, scale=1 / exp_lambda)

        df_hist_sample_2 = df_hist_sample_2.sample(n=sample_size, weights=sample_weights.reshape(-1)).copy()
        df_hist_sample = pd.concat([df_hist_sample_1, df_hist_sample_2], ignore_index=True)

        return df_hist_sample

    @staticmethod
    def _get_expected_values(df_prob, averages):
        '''
        Expected values of the `target_value` will be:

            # non-NAN target_values in that segment-time cell
            -------------------------------------------------   *  np.mean(target_value) for that segment-time cell
                   # rows in that segment-time cell
        '''
        if len(averages.shape) > 1:
            df_sched_expected = df_prob * averages
        else:
            df_sched_expected = df_prob.copy()

            for col in df_sched_expected:
                df_sched_expected.loc[:, col] = averages * df_sched_expected[col]

        return df_sched_expected

    @staticmethod
    def _interpolate_expected_values(df_sched_expected, time_period_drift_len=3):
        # Fill missing segment-time periods with the closest value (left or right), within time_period_drift_len
        print(f"Filling missing {time_period_drift_len}-adjacent time periods with closest available.")
        df_sched_expected_ = df_sched_expected.fillna(method='ffill', axis=1, limit=time_period_drift_len)
        df_sched_expected_ = df_sched_expected_.fillna(method='bfill', axis=1, limit=time_period_drift_len)

        return df_sched_expected_

    def _fill_sim_values(self, df):
        df_ = df.fillna(self.defaults, axis=0)
        df_ = df_.fillna(df.mean(axis=0))
        df_ = df_.replace([np.inf], 2 ** 32 - 1)
        df_ = df_.replace([-np.inf], -2 ** 32 + 1)

        return df_

    def _pca_reduction(self, df_sim, pca_model=None, perc_var=.98):
        '''

        Parameters
        ----------
        df_sim
        pca_model
        perc_var : float
            Minimum percent of variability required. In testing, 99% of variability was captured by 2 of 2K features
        Returns
        -------

        '''
        df_sim_ = self._fill_sim_values(df_sim)

        print(f"Reducing {df_sim_.shape[1]} dimensions ...")
        if pca_model is None:
            pca_model = IncrementalPCA(copy=False,
                                       batch_size=int(np.sqrt(df_sim_.shape[1])))
            X_sim = pca_model.fit_transform(df_sim_.values)
        else:
            X_sim = pca_model.transform(df_sim_.values)

        eigenvalues = pca_model.singular_values_
        eigen_cum_var = np.cumsum(eigenvalues / eigenvalues.sum())  # Cumulative variance % captured
        n_components = len(eigenvalues[eigen_cum_var <= perc_var])
        print(f"\n{perc_var * 100} of variation lies in {n_components} principal components ...")

        return X_sim[:, :n_components], pca_model

    @staticmethod
    def _convert_to_numeric(df_sim):
        print("Converting all data to numerical or dummy (binary indicator) labels.")
        # Get numerical values for the similarity columns to calculate similarity

        # TODO: multiprocessing on this to speed up ... There could be a lot of columns
        for col in df_sim:
            try:
                df_sim.loc[:, col] = df_sim[col].astype(float)
            except:
                df_sim_dummies = pd.get_dummies(df_sim[col], prefix=col)
                df_sim.drop(columns=col, inplace=True)
                df_sim = pd.concat((df_sim, df_sim_dummies), axis=1)

        return df_sim

    def _get_target_neighbors(self, df_sched_expected_, n_neighbors=2):
        agg_funcs = {'mean': lambda x: np.mean(x, axis=1),
                     'min': lambda x: np.min(x, axis=1),
                     'max': lambda x: np.max(x, axis=1),
                     'q25': lambda x: np.nanquantile(x, 0.25, axis=1),
                     'median': lambda x: np.nanquantile(x, 0.5, axis=1),
                     'q75': lambda x: np.nanquantile(x, 0.75, axis=1),
                     'std': lambda x: np.std(x, axis=1),
                     # 'mean_diff': lambda x: np.nanmean(np.diff(x.fillna(method='pad', axis=1), axis=1), axis=1),
                     'count': lambda x: np.sum(~np.isnan(x), axis=1),
                     'sum': lambda x: np.sum(x, axis=1)}

        df_sim = pd.DataFrame()

        print(f"Calculating aggregate statistics for {self.target_value} behavior.")
        for agg in agg_funcs.keys():
            func = agg_funcs[agg]
            df_sim[agg] = func(df_sched_expected_)

        X_sim, _ = self._pca_reduction(df_sim)

        print(f"Computing {self.target_value} nearest neighbors by similarity in aggregate statistics.")
        neighbors = NearestNeighbors(n_neighbors=n_neighbors)
        neighbors.fit(X_sim)
        distances, indices = neighbors.kneighbors(X_sim)

        return distances, indices

    def _get_segment_neighbors(self, df, n_neighbors=2):
        print("Merging data to determine segment index, and collect similarity features.")
        df_sim = df[self.similarity_features + ['segment']].copy()

        df_sim = self._convert_to_numeric(df_sim)

        similarity_features_num = [col for col in df_sim if col != 'segment']
        # Within each segment, use the average value (recall we have binaries) to represent the segment as a whole.
        df_sim_ = df_sim.groupby('segment')[similarity_features_num].mean()
        df_sim_.sort_index(inplace=True)

        print("Reducing dimensionality of segments similarity data.")
        # Reduce the number of columns (dummies make columns explode) using PCA
        X_sim, _ = self._pca_reduction(df_sim_)

        print(f"Computing segments' nearest {n_neighbors} neighbors.")
        neighbors = NearestNeighbors(n_neighbors=n_neighbors)
        neighbors.fit(X_sim)
        distances, indices = neighbors.kneighbors(X_sim)

        return distances, indices

    @staticmethod
    def _fill_by_neighbors(distances, indices, df, distance_lenience=0.9, axis=1):
        '''

        Parameters
        ----------
        distances : np.array
            (Output of NearestNeighbors.kneighbors[0]) Basically a list of lists. The j-th item in the i-th list
            represents the distance between the i-th item (index) in df and the indices[i][j]-th item in df.

        indices : np.array
            (Output of NearestNeighbors.kneighbors[1]) List of lists. The

        df : pd.DataFrame
            The index of this DataFrame must be the *same* as the index of the DataFrame passed into NearestNeighbors.

        distance_lenience : float
            Between 0 and 1 (inclusive). Essentially the quantile of distances that you want to accept when filling
            missing values.

        axis : int
            Which axis matters when comparing original and filled values. Basically, what axis represents the
            features of an entity. Traditionally this will be 1 where each row is an entity.

        Returns
        -------

        '''
        # First get a DataFrame where the index is shared with the DataFrame passed into NearestNeighbors
        df_neighbors = pd.DataFrame(index=df.index, data=indices)

        # The output of the neighbors does not map to the index of the input, just sequential order
        df_neighbors = df_neighbors.astype(np.int64)

        index_map = {k: v for k, v in zip(np.arange(df.shape[0], dtype=np.int64), df.index.values)}

        for col in df_neighbors.columns:
            df_neighbors.loc[:, col] = df_neighbors[col].map(index_map)

        max_distance = distances.max()
        max_distance_allowed = np.quantile(distances, distance_lenience)
        print(f"Maximum distance between points: {round(max_distance, 4)}")
        print(f"Maximum distance allowed (below {distance_lenience * 100}% quantile): {round(max_distance_allowed, 4)}")

        for neighbor in range(indices.shape[1]):
            df_filler = df.copy()
            df_filler = df_filler.loc[df_neighbors[neighbor]]
            df_filler.index = df.index
            df_filler.iloc[distances[:, neighbor] > max_distance_allowed] = np.nan
            df_ = df.fillna(df_filler)

            median_diffs = df_.median(axis=axis) - df.median(axis=axis)
            mean_perc_diff = np.mean(np.abs(median_diffs) / df.median(axis=axis))
            mean_perc_diff = round(mean_perc_diff * 100, 1)

            print(f"Neighbor {neighbor + 1}: Mean Err. and MAPE between last & filled axis-{axis} medians:", end=" ")
            print(f"{round(median_diffs.mean(), 4)} | {mean_perc_diff}%")

            df = df_

            perc_full = (~df.isna()).sum().sum() / \
                        (df.shape[0] * df.shape[1])
            perc_full = np.round(perc_full * 100, 2)

            print(f"Neighbor {neighbor + 1}: Matrix integrity = {perc_full}% full")

        return df

    def _fill_missing_schedule(self,
                               df,
                               df_sched_expected,
                               time_period_drift_len,
                               behavior_neighborhood,
                               segment_neighborhood):

        df.sort_index(inplace=True)
        df_sched_expected.sort_index(inplace=True)

        # Interpolate by values nearby values (time based)
        df_sched_expected_ = self._interpolate_expected_values(df_sched_expected, time_period_drift_len)

        # Fill by target behavior similarity
        distances, indices = self._get_target_neighbors(df_sched_expected, behavior_neighborhood)
        df_sched_expected_ = self._fill_by_neighbors(distances,
                                                     indices,
                                                     df_sched_expected_,
                                                     distance_lenience=0.99)

        # Fill by segment similarity
        df_ = df[df['segment'].isin(df_sched_expected_.index)].copy()
        distances, indices = self._get_segment_neighbors(df_, segment_neighborhood)
        df_sched_expected_ = self._fill_by_neighbors(distances,
                                                     indices,
                                                     df_sched_expected_,
                                                     distance_lenience=0.9)

        return df_sched_expected_

    def _get_default_sim_values(self, df):
        '''
        For each similarity feature, get either the average or the most common value amidst some "main" dataset, df.
        '''
        defaults = {}

        for col in self.similarity_features:
            # Check to ensure that it is numeric, and if so, take the average
            try:
                df.loc[:, col] = df[col].astype(float)
                defaults[col] = df[col].mean()

            # Otherwise, take the most common
            except:
                defaults[col] = df[col].mode().iloc[0]

        self.defaults = pd.Series(defaults)

    def fit(self,
            df,
            time_period,
            best_target_predictor,
            time_period_drift_len=5,
            segment_neighborhood=5,
            behavior_neighborhood=5,
            knn_kws=None):

        self._get_default_sim_values(df)

        self._get_hist_segments(df)
        self._fit_knn(df, knn_kws)

        df_ = df.merge(self.df_segments[self.segment_features + ['segment']],
                       on=self.segment_features,
                       how='left')

        df_prob = self._get_bayes_prob(df_, time_period)

        # df_segment_t_mean = self._get_segment_t_mean(df_, time_period)
        #
        # df_sched_expected = self._get_expected_values(df_prob, df_segment_t_mean)
        #
        # df_sched_expected_ = self._fill_missing_schedule(df_,
        #                                                  df_sched_expected,
        #                                                  time_period_drift_len,
        #                                                  behavior_neighborhood,
        #                                                  segment_neighborhood)

        df_prob_ = self._fill_missing_schedule(df_,
                                               df_prob,
                                               time_period_drift_len,
                                               behavior_neighborhood,
                                               segment_neighborhood)

        df_segment_means = self._get_segment_means(df_, best_target_predictor)

        df_sched_expected_ = self._get_expected_values(df_prob_, df_segment_means)

        self.df_sched_expected = df_sched_expected_

    def _fit_knn(self, df, knn_kws=None, use_means=False):
        '''
        Given a dataset with similarity features (as defined) and segment features (to merge on to get segments),
        we fit a KNN classifier to predict the segment that something is in based on a dimension-reduced isomorphism
        from the similarity features.

        Parameters
        ----------
        df
        knn_kws

        Returns
        -------

        '''
        df_ = df.merge(self.df_segments,
                       on=self.segment_features,
                       how='inner')

        df_sim = df_[self.similarity_features + ['segment']].copy()
        df_sim = self._convert_to_numeric(df_sim)
        self.knn_features = [col for col in df_sim if col != 'segment']

        if use_means:
            df_sim = df_sim.groupby('segment')[self.knn_features].mean().reset_index()

        X_sim, self.knn_pca_model = self._pca_reduction(df_sim[self.knn_features])

        if knn_kws is None:
            knn_kws = {'n_neighbors': 2}

        print("Fitting K Nearest Neighbors ...")
        knn = KNeighborsClassifier(**knn_kws)
        knn.fit(X_sim, df_sim['segment'].values.astype(int))
        self.knn = knn

    @staticmethod
    def get_second_best(a):
        '''
        In this case, "second best" in a list containing a single value *is* that single value.
        '''
        series = pd.Series(a)
        series = series.drop_duplicates().dropna().sort_values(ascending=False)

        if len(series) < 1:
            return np.nan
        elif len(series) < 2:
            return series[0]
        else:
            return series[1]

    def _assign_closest_segment(self, df):
        # Use NCA to assign segment given a fit of known segment data, features are similarity features.
        pass

    def _predict_segments(self, df):
        # First get segments based on actual segment feature values (the combinations of which we've seen)
        df = pd.merge(df, self.df_segments, on=self.segment_features, how='left')
        perc_familiar_segments = round(((~df['segment'].isna()).sum() / df.shape[0]) * 100, 2)

        print(f"{perc_familiar_segments}% of the to-predict rows are in segments we've seen before.")
        if df['segment'].isna().sum() == 0:
            return df

        print("We will use similarity for the rest")

        # Then tease out the ones where we don't have a match
        df_sim = df[df['segment'].isna()][self.similarity_features].copy()
        df_sim = self._convert_to_numeric(df_sim)

        # To predict segments, we first need to get the numeric columns used in the KNN model
        missing_sim_features = []
        for col in self.knn_features:
            if col not in df_sim.columns:
                missing_sim_features.append(col)
                # The only way we have a missing feature column is if it was generated by pd.get_dummies
                # So we fill it with 0 because for that row, it is not the case that it meets that indicator
                df_sim[col] = 0

        new_sim_features = [col for col in df_sim if col not in self.knn_features]

        print(f"\n\nMissing Similarity Features:\n{missing_sim_features}")
        print(f"\nNew Similarity Features:\n{new_sim_features}\n")

        X_sim, _ = self._pca_reduction(df_sim[self.knn_features].copy(), self.knn_pca_model)
        df.loc[df_sim.index, 'segment'] = self.knn.predict(X_sim)

        return df

    def predict(self, df, time_period, ids, return_all=False):
        df = self._predict_segments(df)

        df_ = df[ids + ['segment', time_period]].drop_duplicates().copy()

        # For new time periods that haven't been seen, fill with NA
        for t in df[time_period].unique():
            if t not in self.df_sched_expected.columns:
                self.df_sched_expected[t] = np.nan

        df_['expected_today'] = df_.apply(lambda r: self.df_sched_expected.loc[r['segment'],
                                                                               r[time_period]], axis=1)

        df_['max_expected'] = df_.apply(lambda r: self.df_sched_expected.loc[r['segment'], :].max(), axis=1)

        df_['second_best_expected'] = df_.apply(lambda r:
                                                self.get_second_best(self.df_sched_expected.loc[r['segment'], :]),
                                                axis=1)

        df_['today_is_best'] = df_['expected_today'] == df_['max_expected']

        df_['ranking_value'] = (df_['expected_today'] - df_['second_best_expected']) \
                                        .where(df_['today_is_best'],
                                               other=df_['expected_today'] - df_['max_expected'])

        df_.sort_values('ranking_value', ascending=False, ignore_index=True, inplace=True)

        df_['rank'] = df_.index.where((~df_['ranking_value'].isna()), other=np.nan) + 1

        if return_all:
            return df_
        else:
            return df_[ids + ['rank']]

    def save(self, path):
        '''
        When loading, load the most recent of these.
        '''
        today = pd.to_datetime(pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S'))
        self.df_segments.to_pickle(f'{path}/df_segments_{today}.pkl')
        self.df_sched_expected.to_pickle(f'{path}/df_sched_expected_{today}.pkl')

    @staticmethod
    def update_weighted_average(segment_means_all, segment_means_update,
                                segment_counts_all, segment_counts_update,
                                new_data_weight=1):

        segment_data = pd.DataFrame({'all_means': segment_means_all,
                                     'update_means': segment_means_update})

        segment_data['all_counts'] = segment_counts_all
        segment_data['update_counts'] = segment_counts_update

        segment_means = segment_data[['all_means', 'update_means']].copy()
        segment_counts = segment_data[['all_counts', 'update_counts']].copy()

        segment_means = segment_means.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
        segment_counts = segment_counts.fillna(0)
        segment_counts[segment_counts.sum(axis=1) < 1] = 1

        segment_counts.loc[:, 'update_counts'] = segment_counts['update_counts'] * new_data_weight

        weighted_average = np.average(segment_means, axis=1, weights=segment_counts)

        return pd.Series(data=weighted_average, index=segment_data.index)

    def update(self,
               df_hist,
               event_time,
               time_period,
               best_target_predictor,
               df_segments=None,
               df_sched_expected=None,
               defaults=None,
               n_lookback_days=28 * 6,
               today=None,
               knn_kws=None,
               time_period_drift_len=5,
               behavior_neighborhood=5,
               segment_neighborhood=5,
               new_data_weight=1):

        if defaults is None:
            self._get_default_sim_values(df_hist)
        else:
            self.defaults = defaults

        print("Updating ongoing list of segments with new segments.")
        self.df_segments = df_segments
        self._add_segments(df_hist)
        self._fit_knn(df_hist, knn_kws)

        df_hist_ = df_hist.merge(self.df_segments[self.segment_features + ['segment']],
                                 on=self.segment_features,
                                 how='left')

        df_update = self._get_update_sample(df_hist_,
                                            event_time,
                                            n_lookback_days=n_lookback_days,
                                            today=today)

        df_probs = self._get_bayes_prob(df_hist_, time_period)

        # df_segment_t_mean = self._get_segment_t_mean(df_update, time_period)
        #
        # df_sched_expected_ = self._get_expected_values(df_probs, df_segment_t_mean)

        df_prob_ = self._fill_missing_schedule(df_hist_,
                                               df_probs,
                                               time_period_drift_len,
                                               behavior_neighborhood,
                                               segment_neighborhood)

        df_segment_means_all = self._get_segment_means(df_hist_, best_target_predictor)
        df_segment_means_sample = self._get_segment_means(df_update, best_target_predictor)
        df_segment_counts_all = df_hist_.groupby('segment')[self.target_value].count()
        df_segment_counts_sample = df_update.groupby('segment')[self.target_value].count()

        df_segment_means = self.update_weighted_average(df_segment_means_all,
                                                        df_segment_means_sample,
                                                        df_segment_counts_all,
                                                        df_segment_counts_sample,
                                                        new_data_weight)

        df_sched_expected_ = self._get_expected_values(df_prob_, df_segment_means)

        # mask = df_sched_expected_.index.isin(df_update['segment'].unique())
        # df_sched_expected_ = df_sched_expected_[mask].copy()

        # df_sched_expected_ = self._fill_missing_schedule(df_hist_,
        #                                                  df_sched_expected_,
        #                                                  time_period_drift_len,
        #                                                  behavior_neighborhood,
        #                                                  segment_neighborhood)

        # For new segments, we want to update the value if there is one, keep the old data if there are no updates
        update_index = df_sched_expected_.index
        new_segments = [i for i in update_index if i not in df_sched_expected.index]

        df_sched_expected = pd.concat([df_sched_expected, df_sched_expected_.loc[new_segments]])

        # TODO: Consider a different scheme here, think about observation count in the segment
        df_sched_expected.loc[update_index, :] = df_sched_expected_.where((~df_sched_expected_.isna()),
                                                                          df_sched_expected.loc[update_index, :])

        self.df_sched_expected = df_sched_expected
