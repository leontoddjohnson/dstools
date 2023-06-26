import numpy as np


class Slicer(object):
    def __init__(self, features, value='meets_preference', agg_list=None):
        '''
        Set (preferably) categorical features, and given a dataframe, use these features to build segments, and
        for each segment, collect aggregate information on some value.

        This is essentially a faster way to GroupBy for multiple columns.

        Parameters
        ----------
        features : list of str
            columns to use for segmentation
        value : str
            numerical column of interest
        agg_list : list of tuples
            [(agg_name, agg), ...]
        '''

        self.features = features
        self.value = value
        self.agg_list = agg_list if agg_list is not None else list()
        self.df_segments = None

        if ('num_in_segment', 'count') not in self.agg_list:
            self.agg_list.append(('num_in_segment', 'count'))

    def get_segments(self, df, segment_name='segment'):
        df_key = df[self.features].drop_duplicates().copy().reset_index(drop=True)
        df_key[segment_name] = df_key.index
        df = df.merge(df_key, on=self.features, how='left')

        df.loc[:, self.value] = df[self.value].astype(float)
        df_segments = df.groupby(segment_name)[self.value] \
            .agg(self.agg_list).reset_index()

        self.df_segments = df_segments.merge(df_key, on=segment_name, how='left')

        return self.df_segments

    def get_segments_slice(self, df, min_in_group=50, mask=None, round=2):
        if mask is None:
            mask = [True] * df.shape[0]

        df_slice = self.get_segments(df[mask]).copy()
        df_slice = df_slice[df_slice['num_in_segment'] >= min_in_group]

        for agg in self.agg_list:
            p_group = df[mask][self.value].apply(agg[1])
            df_slice[f'{agg[0]}_perc_diff'] = (((df_slice[agg[0]] - p_group) / p_group) * 100).round(round)
            print(f"{agg[0]} of '{self.value}' for the whole group in this mask: ", np.round(p_group, round))

        return df_slice
