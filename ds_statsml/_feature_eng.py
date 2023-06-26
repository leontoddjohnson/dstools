import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from pandas.tseries.holiday import USFederalHolidayCalendar, Holiday, SU, MO, TU, WE, TH, FR, SA
from pandas.tseries.offsets import DateOffset
from sklearn.utils import column_or_1d
from ._timeseriestools import get_fourier, get_consecutive
from dstools.ds_util import define_bins
from multiprocessing import current_process


def get_timeseries_features(df_all, timeseriescols, timeseriescols_diffs=None,
                            groupcol='ASIN', datecol='Date', window_back=2, window_forward=2, agg_dict=None, new_aggs=None,
                            get_fourier_values=False, pool=None):

    if timeseriescols_diffs is None:
        timeseriescols_diffs = []

    print('(Getting Time Series) Marking consecutive rows with enough data ...')
    # Mark the rows of the data that fall into series with the `groupcol` and and the `datecol`
    df_all = get_consecutive(df=df_all,
                             groupcol=groupcol,
                             datecol=datecol,
                             consec_seq_length=window_back + window_forward + 1,
                             agg_dict=agg_dict,
                             new_aggs=new_aggs,
                             reset_index=True)

    df_all.rename(columns={f'{window_back + window_forward + 1}_day_seq': 'seq'}, inplace=True)

    # Given rows that fall consecutively, get the indices to shift on
    data_shifts = df_all[[groupcol, 'seq']].copy()
    data_shifts['_0'] = df_all.index

    print('(Getting Time Series) Annotate windows ...')
    for i in range(window_back, 0, -1):
        data_shifts.loc[:, f'_-{i}'] = data_shifts.groupby([groupcol, 'seq'])['_0'].shift(-i)

    for i in range(1, window_forward + 1):
        data_shifts.loc[:, f'_+{i}'] = data_shifts.groupby([groupcol, 'seq'])['_0'].shift(i)

    data_shifts.dropna(inplace=True)
    windows = [f'_-{i}' for i in range(1, window_back + 1)][::-1] + ['_0'] + [f'_+{i}' for i in range(1, window_forward + 1)]
    data_shifts = data_shifts[windows]

    # For defined timeseriescols, get the time series (using the shifted data index), and aggregate avg, max, and min
    if pool is not None:
        shift_results = [pool.apply_async(calculate_data_shift,
                                          (df_all[col], col, windows, data_shifts, col in timeseriescols_diffs, window_back, window_forward))
                         for col in timeseriescols]

        for res in shift_results:
            data_shifts = data_shifts.join(res.get())

    else:
        for col in timeseriescols:
            shifted_data = calculate_data_shift(df_all[col], col, windows, data_shifts, col in timeseriescols_diffs, window_back, window_forward)
            data_shifts = data_shifts.join(shifted_data)

    data_shifts.drop(columns=windows, inplace=True)

    # Here is where we get rid of the rows that do not have sufficient data for all the time series values
    df_all = df_all.join(data_shifts, how='right')

    # The timeseriescols will be replaced with `timeseriescol_0` from the windows operation
    df_all.drop(columns=timeseriescols, inplace=True)

    for tscol in timeseriescols:
        tscol_windows = [tscol + window for window in windows]

        if get_fourier_values:
            amps = get_fourier(df_all[tscol_windows].values)[0]
            phases = get_fourier(df_all[tscol_windows].values)[1]

            ts_amps = pd.DataFrame(index=df_all.index, columns=[tscol + '_fourier_amp' + window for window in windows], data=amps)
            ts_phases = pd.DataFrame(index=df_all.index, columns=[tscol + '_fourier_phs' + window for window in windows], data=phases)

            df_all = pd.concat((df_all, ts_amps, ts_phases), axis=1)

    return df_all


def calculate_data_shift(df_col, col, windows, data_shifts, is_timeseries_diff, window_back, window_forward):
    print(f"[{current_process().pid}] Getting time series split for {col} ...")
    windowcols = [col + w for w in windows]
    shifted_data = np.vstack([df_col[data_shifts[window]].values for window in windows]).T
    shifted_data = pd.DataFrame(index=data_shifts.index,
                                columns=windowcols,
                                data=shifted_data)

    shifted_data.loc[:, col + '_avg'] = shifted_data[windowcols].mean(axis=1)
    shifted_data.loc[:, col + '_max'] = shifted_data[windowcols].max(axis=1)
    shifted_data.loc[:, col + '_min'] = shifted_data[windowcols].min(axis=1)

    if is_timeseries_diff:
        # If we also want the time series of differences, gather the same data, and aggregate
        diff_cols = [col + f'_diff_{i}-{abs(i - 1)}' for i in range(-window_back + 1, window_forward + 1)]

        shifted_data_ = pd.DataFrame(index=shifted_data.index,
                                     columns=diff_cols,
                                     data=shifted_data[windowcols].iloc[:, 1:].values - shifted_data[windowcols].iloc[:, :-1].values)

        shifted_data = shifted_data.join(shifted_data_)

        shifted_data.loc[:, col + '_avg_diff'] = shifted_data[diff_cols].mean(axis=1)
        shifted_data.loc[:, col + '_max_inc'] = shifted_data[diff_cols].max(axis=1)
        shifted_data.loc[:, col + '_max_dec'] = shifted_data[diff_cols].min(axis=1)
        shifted_data.loc[:, col + '_min_abs_diff'] = shifted_data[diff_cols].abs().min(axis=1)

    return shifted_data


def id2cols(df, col, max_len=None, inplace=True, padside='right', fill_pad=-1):
    '''
    Change alphanumeric IDs (*not* case sensitive) to columns (one column per 'digit'), and pad the end of the ID for the longest ID in the column.
    '''
    if not inplace:
        df_ = pd.DataFrame({col: df[col].values})
    else:
        df_ = df

    if max_len is None:
        max_len = np.max(df_[col].apply(len))

    # Use a nonalphanumeric character to generate a Value Error, allowing us to fill with `fill_pad`
    df_.loc[:, f'{col}_padded'] = df_[col].str.pad(max_len, side=padside, fillchar='*')

    try:
        df_[[f'{col}{i}' for i in range(max_len)]] = df_[f'{col}_padded']\
                                                         .str.extractall('(.)', flags=re.U)[0]\
                                                         .unstack().rename_axis(mapper=None, index=1).iloc[:, :max_len]
    # For alternate pandas version compatibility
    except TypeError:
        df_[[f'{col}{i}' for i in range(max_len)]] = df_[f'{col}_padded'] \
                                                         .str.extractall('(.)', flags=re.U)[0] \
                                                         .unstack().rename_axis(None, 1).iloc[:, :max_len]

    def id2int(x):
        try:
            return int(x, 36)
        except ValueError:
            return fill_pad

    for i in range(max_len):
        df_.loc[:, f'{col}{i}'] = df_[f'{col}{i}'].apply(id2int)

    if not inplace:
        return df_


# SKLearn Wrappers
class SafeLabelEncoder(LabelEncoder):
    """An extension of LabelEncoder that will
    not throw an exception for unseen data, but will
    instead return a default value of 999999

    Attributes
    ----------

    classes_ : the classes that are encoded
    """

    def __init__(self, max_nlabels=999999):

        self.max_nlabels = max_nlabels

    def _get_unseen(self):
        """Basically just a static method
        instead of a class attribute to avoid
        someone accidentally changing it."""
        return self.max_nlabels

    def transform(self, y):
        """Perform encoding if already fit.

        Parameters
        ----------

        y : array_like, shape=(n_samples,)
            The array to encode

        Returns
        -------

        e : array_like, shape=(n_samples,)
            The encoded array
        """

        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)

        classes = np.unique(y)

        # Check not too many:
        unseen = self._get_unseen()
        if len(classes) >= unseen:
            raise ValueError('Too many factor levels in feature. Max is %i' % unseen)

        e = np.array([np.searchsorted(self.classes_, x) if x in self.classes_ else unseen for x in y])

        return e


# This needs to be adjusted to work for data with minimum values greater than 0
def getbucketbins(data, targetcol, bucket_col_name='SalesDivision', leftover_cutoff=2000, buckets=None, buckets_per_oom=(1, 2, 5)):
    buckets_per_oom = sorted([b for b in buckets_per_oom if 0 < b < 10])

    if buckets is None:
        max_value = data[targetcol].max()
        max_oom = np.floor(np.log10(max_value)).astype(int)

        # Build all buckets for each of the groups we'll classify for
        buckets = [0] + buckets_per_oom
        for oom in range(1, max_oom + 1):
            for thresh in buckets_per_oom:
                buckets.append(thresh * (10 ** oom))

        buckets = np.array(buckets)
        data.loc[:, bucket_col_name] = define_bins(data[targetcol].values.copy(), buckets)
        bucket_counts = data[bucket_col_name].value_counts().sort_index()

        # Go through each bucket (from the largest to the smallest index)
        # Keep a log of what would be contained in the last bucket, and if that container is less than the leftover_cuttoff,
        # bucket it in with the next up
        sum_ = 0
        lastbucket = buckets[-1]

        for bucket, bucket_count in zip(bucket_counts[::-1].index, bucket_counts[::-1]):
            sum_ += bucket_count
            if sum_ < leftover_cutoff:
                continue
            else:
                lastbucket = bucket
                break

        # Redifine the buckets based on the last bucket decided, and save it to the dataframe in place
        buckets = buckets[np.where(buckets <= lastbucket)]

    data.loc[:, bucket_col_name] = define_bins(data[targetcol].values.copy(), buckets)

    # For each bucket, we define a binsize. Each bucket is broken up into bins based on its tenth (not including bucket 0, 1, 10, and the last one)
    binsizes = []

    for bucket in buckets:
        if bucket == 0:
            binsize = 0
        elif bucket == 1:
            binsize = 1
        elif bucket == 10:
            binsize = 2
        else:
            binsize = int(bucket * 0.1)

        binsizes.append(binsize)

    bucketbins = {k: v for k, v in zip(buckets, binsizes)}

    return bucketbins


def getholidays(dates, holiday_window=(5, 5), custom_holidays=None):
    '''
    GET THE NUMBER OF HOLIDAYS WITHIN WINDOW

    Just count the number of holidays (for each row) within a window of the day we're predicting on (see config to change).

    Holidays included: New Years Day, Dr. Martin Luther King Jr. Day, Presidents Day, MemorialDay, July 4th, Labor Day, Columbus Day, Veterans Day,
    Thanksgiving, Christmas, Black Friday, Halloween

    Parameters
    ----------
    dates : array-like
        The dates you'd like to find the number of holidays around.
    holiday_window : tuple
        (<window back>, <window forward). So, this will count the number of holidays between the dates
        <date> - holiday_window[0] and <date> + holiday_window[1]
    custom_holidays : dict of dicts
        A dictionary with the name of the holiday and information on that holiday. Possible holiday date information are: day, month, year,
        offset_weekday, offset_num_weeks. These must be keys in the dictionaries below the names.

        So, for example, you could have

        custom_holidays = {'birthday': dict(month=8, day=14),
                           'Black Friday': dict(month=11, offset_weekday='Friday', offset_num_weeks=4)}

        The `offset_weekday` must be the text of the day (e.g., 'Fri', 'Tuesday', 'WED', etc.)

    Returns
    -------

    '''
    dates = np.array(dates)
    day_map = {k: v for k, v in zip(['sun', 'mon', 'tues', 'wednes', 'wed', 'thurs', 'fri', 'sat'], [SU, MO, TU, WE, WE, TH, FR, SA])}
    default = {'day': None,
               'month': None,
               'year': None,
               'offset_weekday': None,
               'offset_num_weeks': None}

    custom_holidays_ = []

    for name in custom_holidays.keys():
        custom_holiday = {**default, **custom_holidays[name]}

        if custom_holiday['offset_weekday'] is not None and custom_holiday['offset_num_weeks'] is not None:
            day_func = day_map[custom_holiday['offset_weekday'].lower().replace('day', '')]
            offset = DateOffset(weekday=day_func(custom_holiday['offset_num_weeks']))
            day = 1
        else:
            offset = None
            day = custom_holiday['day']

        custom_holiday_ = Holiday(name,
                                  day=day,
                                  month=custom_holiday['month'],
                                  year=custom_holiday['year'],
                                  offset=offset)

        custom_holidays_.append(custom_holiday_)

    # The USFederal Holiday Calendar doesn't include Black Friday, Halloween, or the prime days (of course)
    class AMZCalendar(USFederalHolidayCalendar):
        rules = USFederalHolidayCalendar.rules + \
                [Holiday('Black Friday', month=11, day=1, offset=DateOffset(weekday=FR(4)))] + \
                [Holiday('Halloween', month=10, day=31)] + \
                custom_holidays_

    cal = AMZCalendar()

    # # We find holidays by creating a <n_samples> x (<holiday_window[0]> + <holiday_window[1]>) * <n_holidays> matrix,
    # # where each day in the holiday window is checked among all the n_holidays for matching,
    # # then we sum to find out how many made matches in the window

    # First, we need to get a list of all the holidays within all possible windows
    minday = dates.min() - pd.Timedelta(days=holiday_window[0] + 1)
    maxday = dates.max() + pd.Timedelta(days=holiday_window[1])
    holidays = cal.holidays(start=minday, end=maxday, return_name=True).index

    # Get the last day in the window (centered around the date)
    window_ends = dates.values[:, np.newaxis] + pd.Timedelta(days=holiday_window[1])

    # Build an array where each row represents a window for for all possible dates to check
    windows = window_ends
    for ndays in range(1, holiday_window[0] + 1 + holiday_window[1]):
        windows = np.hstack((window_ends - pd.Timedelta(days=ndays), windows))

    # *For each column* in windows (i.e., each day in the window for each sample), check if it matches any of the <holidays>
    # making an <n_samples> x <n_holidays> matrix. Sum each row in this matrix, and you get a 1 if it matches one (i.e., it's a holiday).
    # Do this for each day in the window, stack them (vertically, because they're 1-D arrays), and then sum them up for the final count.
    isholiday = np.sum(windows[:, 0][:, np.newaxis] == np.array(holidays), axis=1)

    for i in range(1, holiday_window[0] + 1 + holiday_window[1]):
        next_isholiday = np.sum(windows[:, i][:, np.newaxis] == np.array(holidays), axis=1)
        isholiday = np.vstack((isholiday, next_isholiday))

    numholidays = np.sum(isholiday, axis=0)

    return numholidays