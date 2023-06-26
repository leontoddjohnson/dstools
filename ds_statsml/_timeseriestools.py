import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from ._alt_metrics import *
from ._stattools import StatsmodelSKLearn
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, median_absolute_error, \
    recall_score, f1_score, precision_score, accuracy_score


sns.set_style('darkgrid')
sns.set(rc={'patch.edgecolor': 'w', 'patch.force_edgecolor': True, 'patch.linewidth': 1})


def plot_fourier(df, series=4, m=4, freqs=None, scale_amp=True, figsize=(10, 10)):
    '''
    ...

    -------------
    Params
    -------------
    :param df:
    :param series: This is the *index* value for the row in the data that you'd like to look at.
    :param m:
    :param freqs: ({None, list or arraylike}), If None, only plot the top m sinusoids (where 'top' is defined by
    amplitude. Otherwise, plot the frequency indices listed here.
    :param scale_amp: (bool), If True, scale the amplitudes of the components to plot where the most influential can
    only have an amplitude of the maximum value of the series, and the minimum can be 0. This just makes it easier to see
    the influence of the components on the original series. However, if True, you cannot interpret the amplitudes as
    they are.
    :param figsize: Typical ...
    -------------
    Return
    -------------
    :return:
    '''

    series_vals = df.iloc[series].values
    N = len(series_vals)

    # Get a list of tuples, with information for each of the frequency indices from 0 to N-1 for the series chosen
    A_series_freqs = get_fourier(series_vals)

    if freqs:
        # Get just specific frequency info for the frequencies/seasons requested
        A_freqs = [x for x in A_series_freqs if x[0] in freqs]
    else:
        # Sort the frequencies (by most influential, or highest amplitude), and take the top m
        A_freqs = sorted(list(A_series_freqs), key=lambda x: x[1], reverse=True)[:m]

    # In case we need to scale the amplitudes of the components, we take them from 0 to the maximum series value
    amps = [x[1] for x in A_freqs]
    amps_scaled = minmax_scale(amps, (0, series_vals.max()))

    # Plot Series
    fig, axes = plt.subplots(2, 1)
    fig.set_size_inches(figsize[0], figsize[1])
    fig.suptitle('Fourier Contributions', y=0.91, fontsize=16)

    # Get some x values to plot
    time = np.arange(0, N, 0.1)

    for i, info in enumerate(A_freqs):
        freq = info[0]
        if scale_amp:
            amp = amps_scaled[i]
        else:
            amp = info[1]
        phase = info[2]

        # Value on the cosine wave of the freq-th frequency sinusoid
        cosine = amp * np.cos(2 * np.math.pi * freq * time / N + phase)

        # Plot a cosine wave for this frequency
        axes[1].plot(time, cosine, label=f'Freq: {freq}, True amp: {np.round(info[1], 2)}', alpha=0.7)

    axes[1].plot(series_vals, label=f'Orig Series', ls='--', color='blue')
    sns.lineplot(np.array(range(N)), series_vals, ax=axes[0], label='Orig Series')
    sns.regplot(np.array(range(N)), series_vals, ax=axes[0], fit_reg=True, lowess=True, scatter_kws={'s': 0},
                line_kws={'color': 'red',
                          'ls': '--',
                          'label': 'LOWESS Fit'})
    axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.8))
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.8))

    return fig, axes


def get_fourier(a, axis=-1):

    a = np.array(a)
    N = a.shape[-1]

    # Then, we'll get the coefficients in the transformed Fourier Space, these are complex values (e.g., a + ib)
    A_full = np.fft.fft(a, axis=axis)

    # The magnitudes of A_full are the amplitudes, and the phase (offset) is the angle of the complex value
    A_amp = np.abs(A_full)
    A_phase = np.angle(A_full)

    if len(a.shape) == 1:
        A_series_freqs = [(freq, amp, phase) for freq, amp, phase in zip(range(N), A_amp, A_phase)]
        return A_series_freqs

    else:
        return A_amp, A_phase


def tss_gridsearch(X, y, model_instance, param_grid=None, n_splits=5, overlap=0.1, scores=None, statsmodel=False,
                   statsmodel_reg=False, return_train_score=True, train_size=0.8, num_jobs=4):
    '''
    **X and y MUST BE SORTED BY DATE IN ASCENDING ORDER FIRST!!**

    This will do a grid search in Time Series fashion. The default functionality for this method is to run a grid search on the training data,
    and do evaluation on the training data.

    To get a dataframe of the grid search results, run:

    grid_df = pd.DataFrame(grid_est.cv_results_)

    Parameters:

    :param X: np.array, The features to model on
    :param y: np.array, The target values to do the grid search on
    :param model_instance: {statsmodel, lgb.sklearn.LGBMClassifier, lgb.sklearn.LGBMRegressor
    :param n_splits:
    :param overlap: float in [0, 1],
    :param scores: ... the first of these will be the 'refit' score in sklearn.model_selection.GridSearchCV
    :param statsmodel:
    :param param_grid:
    :param statsmodel_reg:
    :param return_train_score:
    :param train_size:
    :param num_jobs:
    :return:
    '''
    if not param_grid:
        param_grid = {}

    # ... You can always add to this for other types of analysis.
    scoring_all = {'R2': make_scorer(r2_score),
                   'RMSE': make_scorer(RMSE),
                   'MAE': make_scorer(mean_absolute_error),
                   'MdAE': make_scorer(median_absolute_error),
                   'MAPE': make_scorer(MAPE),
                   'sMAPE': make_scorer(sMAPE),
                   'MAAPE': make_scorer(MAAPE),
                   'recall': make_scorer(recall_score),
                   'f1_score': make_scorer(f1_score),
                   'precision': make_scorer(precision_score),
                   'accuracy': make_scorer(accuracy_score),
                   'MAAPE_inv': make_scorer(MAAPE_inv)}

    if not scores:
        scores = ['MAAPE', 'RMSE']

    scoring = {score: scoring_all[score] for score in scores}

    tss = TimeSeriesSplit(n_splits, max_train_size=int(X.shape[0] / (overlap * (1 - n_splits) + n_splits)))

    if statsmodel:
        # SKLearn wrapper for statsmodels.
        model = StatsmodelSKLearn(model_instance, regularize=statsmodel_reg)
    else:
        model = model_instance

    grid_est = GridSearchCV(model, param_grid=param_grid, cv=tss, return_train_score=return_train_score, scoring=scoring,
                            refit=scores[0], error_score=np.nan, iid=False, verbose=2, n_jobs=num_jobs)

    # This is where we set the training and test sets for the GridSearch.
    # The test set will happen after the training set, to simulate testing on new, unseen data
    splitpoint = int(X.shape[0] * train_size)

    X_train, y_train = X[:splitpoint], y[:splitpoint]
    # X_test, y_test = X[splitpoint:], y[splitpoint:]

    # if isinstance(model_instance, lgb.sklearn.LGBMClassifier) or isinstance(model_instance, lgb.sklearn.LGBMRegressor):
    #     if 'early_stopping_round' in param_grid.keys():
    #         fit_params = {'eval_set': [(X_test, y_test)], 'eval_names': ['Eval']}
    #     else:
    #         fit_params = {}
    # else:
    #     fit_params = {}

    fit_params = {}

    grid_est.fit(X_train, y_train, **fit_params)

    return grid_est


def TS_outlier_to_na(ts, diff_outlier_factor=1.5):
    '''

    :param ts: np.array, The time series (in order)
    :param diff_outlier_factor: {int, float}, A factor, in standard deviations. The standard is 1.5 std deviations above or below the mean
    of differences
    :return: np.array, Outliers are now np.nan
    '''
    x = ts[~np.isnan(ts)].astype(float)

    diffs1 = x[1:] - x[:-1]
    diffs2 = x[2:] - x[:-2]

    mask1 = (diffs1 > (diffs1.mean() + diff_outlier_factor * diffs1.std())) | (diffs1 < (diffs1.mean() - diff_outlier_factor * diffs1.std()))
    mask2 = (diffs2 > (diffs2.mean() + diff_outlier_factor * diffs2.std())) | (diffs2 < (diffs2.mean() - diff_outlier_factor * diffs2.std()))

    x[np.insert(mask1 & np.insert(mask2, 0, False), 0, False)] = np.nan

    ts_new = ts.copy()

    ts_new[np.where(~np.isnan(ts))] = x

    return ts_new


def fill_ts(keyword, timeseries, fname, diff_outlier_factor=1.5, savedir='./impression_TS'):
    ts = TS_outlier_to_na(timeseries, diff_outlier_factor)

    pd.Series(data=ts).to_csv(savedir + "/" + fname + '.csv')
    os.system(f"Rscript --vanilla impute_timeseries.R '{fname}' '{savedir}'")
    series_imputed = pd.read_csv(f'./impression_TS/{keyword}_imputed.csv')
    series_imputed = series_imputed['x']

    return series_imputed


def get_consecutive(df, groupcol='ASIN', datecol='Date', consec_seq_length=3, agg_dict=None, new_aggs=None, reset_index=False, lenience=0):
    '''
    Given a Pandas Dataframe, a "group column" and a date column, go through each of the groups in the group column,
    and determine the sequences that last longer than the consec_seq_length. There should be multiple rows with the same values in the group column,
    but for a given day, these should be group (i.e., in one day, you shouldn't have more than one instance of the `groupcol` value).

    This will give the original dataframe a few new columns:

        n_day_sequence: (For n = `consec_seq_length`), this is just an indicator of which (arbitrary, ish) sequence the row belongs to. This is an
        indicator of the labeled sequence that the row belongs to. A sequence changes if you go over the consec_seq_length, or if you change the
        groupcol. E.g., if the consec_seq_length is 3, you could have

            group    Date         seq
            1        12-01-2018     1
            1        12-02-2018     1
            1        12-12-2018     2
            2        12-09-2018     3
            2        12-10-2018     3

        over_m_days: (For m = `consec_seq_length`-1), this is a boolean value for whether that row belongs to a sequence with at least
        consec_seq_length
        days in a row, consecutively.

        [agg]: For some aggregate function (default: sum), this is the value for the past n days, and NAN if the sequence doesn't have n days of
        history. Note: **This includes the "current day".

    Parameters:

    :param df:
    :param groupcol:
    :param datecol:
    :param consec_seq_length:
    :param agg_dict: This should be in the form {'colname1': 'sum', 'colname2': ['median', 'max]}.
                        You can use 'sum', 'median', 'mean', 'std', 'min', 'argmin', 'max', or 'argmax'
    :param new_aggs: If you want to add agg function(s), use this as a dictionary of new aggregate functions: {'newagg_name': lambda function}. The
    'aggname' here, would be referenced in the agg_dict after a certain column. E.g., {'col': 'newagg_name'}.

        The function should be a lambda function that has an 'axis' parameter. For example, to clip (based on an IQR threshold) and then aggregate:

        IQR_thresh = 3

        corr_sum = lambda a, axis: np.mean(np.clip(a,
                (np.percentile(a, 25, axis=axis) - IQR_thresh * (np.percentile(a, 75, axis=axis) - np.percentile(a, 25, axis=axis)))[:, np.newaxis],
                (np.percentile(a, 25, axis=axis) + IQR_thresh * (np.percentile(a, 75, axis=axis) - np.percentile(a, 25, axis=axis)))[:, np.newaxis]),
                axis=axis) * 30

    :param reset_index: Keep the original index of the dataframe, or reset at the end
    :param lenience: How many days in-between rows will we allow in one sequence. (Default = 0 days, so, contiguous)

    :return:
    '''

    aggfuncs = {'sum': np.sum,
                'median': np.median,
                'mean': np.mean,
                'std': np.std,
                'min': np.min,
                'max': np.max,
                'argmin': np.argmin,
                'argmax': np.argmax}

    if agg_dict is None:
        calc_agg = False
    else:
        calc_agg = True

    if new_aggs is not None:
        aggfuncs = {**aggfuncs, **new_aggs}

    day1 = df[datecol].iloc[0]

    if not hasattr(day1, 'month'):
        df.loc[:, datecol] = pd.to_datetime(df[datecol])

    df.sort_values([groupcol, datecol], ascending=[True, False], inplace=True)

    if reset_index:
        df.reset_index(drop=True, inplace=True)

    df_ = df[[groupcol, datecol]].copy()
    df_.loc[:, 'DayPre'] = df_.groupby([groupcol])[datecol].shift(-1)
    df_.loc[:, 'DayDiff'] = (df_[datecol] - df_.DayPre).apply(lambda x: x.days)

    # This function finds consecutive 1s, and assigns a 'sequence number' to that sequence of ones, and updates the number as it moves through
    global j
    j = 1

    def get_seq(a):
        global j

        if a <= lenience + 1:
            return j
        else:
            j += 1
            return j - 1

    # Define sequences of consecutive dates, and find the ones that are greater than or equal to the minimum sequence length (including the day of)
    df_.loc[:, 'seq'] = df_.DayDiff.apply(get_seq)
    seq_counts = df_.groupby([groupcol, 'seq'])[datecol].count().reset_index().rename(columns={datecol: 'num_in_seq'})

    seq_counts.loc[:, 'keep'] = seq_counts.num_in_seq > consec_seq_length - 1
    to_keep = seq_counts[seq_counts.keep]
    df.loc[:, f'over_{consec_seq_length - 1}_days'] = (df_[groupcol].isin(to_keep[groupcol])) & (df_.seq.isin(to_keep.seq))
    df.loc[:, f'{consec_seq_length}_day_seq'] = df_.seq

    # Calculate some aggregate function for the days with enough past days over the minimum sequence length, and NAN for rows if there are not
    # enough prior
    if calc_agg:
        for valuecol in agg_dict:

            aggnames = agg_dict[valuecol]

            if isinstance(aggnames, str):
                aggnames = [aggnames]

            for aggname in aggnames:
                aggfunc = aggfuncs[aggname]

                df_[f'{valuecol}_m0'] = df[f'{valuecol}']

                for i in range(1, consec_seq_length):
                    df_.loc[:, f'{valuecol}_m{i}'] = df_.groupby([groupcol, 'seq'])[f'{valuecol}_m{i - 1}'].shift(-1)

                # Only look at aggregating if there are enough data for the past days
                df_agg = df_.dropna(subset=[f'{valuecol}_m{x}' for x in range(consec_seq_length)]).copy()
                aggtitle = valuecol + '_' + str(consec_seq_length) + 'day_' + aggname

                df_agg[aggtitle] = aggfunc(df_agg[[f'{valuecol}_m{x}' for x in range(consec_seq_length)]].values, axis=1)

                df.loc[:, aggtitle] = df_agg[aggtitle]

    return df