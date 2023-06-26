import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from itertools import combinations
from multiprocessing import current_process, Pool
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor


class StatsmodelSKLearn(BaseEstimator, RegressorMixin):
    '''
    Usage:
    # create a model
    >> clf = StatsmodelSKLearn(sm.OLS)

    # Print cross val score on this model
    >> print('crossval', cross_val_score(clf, sm.add_constant(ccard.data[['AGE', 'INCOME', 'INCOMESQ']]), ccard.data['AVGEXP'], cv=5))

    '''
    def __init__(self, sm_model, regularize=False):
        self.sm_model = sm_model
        self.model = None
        self.result = None
        self.regularize = regularize

    def fit(self, X, y):

        self.model = self.sm_model(y, X)

        if not self.regularize:
            self.result = self.model.fit()
        else:
            self.result = self.model.fit_regularized()

    def predict(self, X):
        return self.result.predict(X)


def inv_boxcox(a, lam):
    return np.exp(lam ** -1 * np.log(a * lam + 1)) - 1


class ReduceVIF(BaseEstimator, TransformerMixin):
    '''
    Use:
    transformer = ReduceVIF()
    # Only use 10 columns for speed in this example
    X_vif = transformer.fit_transform(X)
    X_vif.head()

    CREDIT:
    Original Author: Roberto Ruiz
    Reference: https://www.kaggle.com/robertoruiz/sberbank-russian-housing-market/dealing-with-multicollinearity

    I made only slight changes in the calculate_vif function to print what I wanted (what makes sense), changed some
    variable names, and added a few comments. Also, there was reference to a 'y' sort of target variable, which isn't
    needed.
    LTJDS, 18 July 2018.
    '''

    def __init__(self, thresh=5.0, impute=True, impute_strategy='median'):
        # From looking at documentation, values between 5 and 10 are "okay".
        # Above 10 is too high and so should be removed.
        self.thresh = thresh

        # The statsmodel function will fail with NaN values, as such we have to impute them.
        # By default we impute using the median value.
        # This imputation could be taken out and added as part of an sklearn Pipeline.
        if impute:
            self.imputer = SimpleImputer(strategy=impute_strategy)

    def fit(self, X):
        print('ReduceVIF fit')
        if hasattr(self, 'imputer'):
            self.imputer.fit(X)
        return self

    def transform(self, X):
        print('ReduceVIF transform')
        columns = X.columns.tolist()
        if hasattr(self, 'imputer'):
            X = pd.DataFrame(self.imputer.transform(X), columns=columns)
        return ReduceVIF.calculate_vif(X, self.thresh)

    @staticmethod
    def calculate_vif(X, thresh=5.0):
        '''
        These are my personal edits. We just want this to print the max VIF values as it takes out features.
        :param X: pd.DataFrame, The X DataFrame
        :param thresh: Lowest accepted VIF to go through.
        :return:
        '''

        assert isinstance(X, pd.DataFrame)
        # Taken from https://stats.stackexchange.com/a/253620/53565 and modified
        # Each time we loop through, we drop a variable, and we need to check the next iteration
        checknext = True

        while checknext:
            variables = X.columns
            checknext = False

            # Run the variance_inflation_factor function on all the factors left over, and find the max
            # First parameter is the whole matrix (left), second is the variable in question
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]
            max_vif = max(vif)

            # If the maximum vif is larger than the threshold, take that factor out of the dataframe, and try again
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                print(f'Current maximum VIF is {X.columns[maxloc]} .... VIF={max_vif}')
                print(f'Dropping {X.columns[maxloc]}.')
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                checknext = True

        return X


def stepwise_selection(X, y,
                       initial_list=(),
                       threshold_in=0.05,
                       threshold_out=0.05,
                       verbose=True):
    """ Perform a forward-backward feature selection
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)

    while True:
        changed = False

        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)

        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed = True

            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()

        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty

        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)

            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:
            break

    return included


def get_outliers(df_all, groupcols, group_abbrev, statcols, outlier_factors=1.5, keep_agg=False, inplace=True):
    '''
    Determine if values in `statcols` are outliers, based on some grouping scheme. I.e., if the row is in <this> group, is it drastically different
    from other items in that group?

    :param df_all:
    :param groupcols: (list) List of columns representing the groupings
    :param group_abbrev: (str) This is the abbreviation you'd like to use in naming columns
    :param statcols: (list) the column names of the values that you'd like to determine outliers
    :param outlier_factors: (str, float) The value to multiply the IQR range to determine outleirs. I.e., we say an outlier is an outlier if

        value > 75th percentile + `outlier_factor` * IQR
                             OR
        value < 25th percentile - `outlier_factor` * IQR

        and we do this for each of the statcols. outlier_factors is either the same length as statcols, or it's a float; the same value for all
        statcols
    :param keep_agg: (bool) Do you want to keep the aggregate values like mean, median, max, min, 75thperc, 25thperc?
    :param inplace: Inplace

    :return:
        df. This is done inplace right now.
    '''

    if inplace:
        df = df_all
    else:
        df = df_all.copy()

    if not hasattr(outlier_factors, 'len'):
        outlier_factors = [outlier_factors] * len(statcols)

    inflated_values = []

    if np.percentile(df[statcols[0]], 25) == np.percentile(df[statcols[0]], 75):
        inflated_value = np.percentile(df[statcols[0]], 25)
        print(f"{statcols[0]} is inflated at the value {inflated_value} (comprises >= 50% of the data).")
        print("Only checking outliers for non-inflated values.")
        inflated_values.append(inflated_value)
    else:
        inflated_values.append(np.nan)

    df_aggdata = df.groupby(groupcols)[statcols[0]] \
        .agg([(group_abbrev + "_mean_" + statcols[0], 'mean'),
              (group_abbrev + "_median_" + statcols[0], 'median'),
              (group_abbrev + "_max_" + statcols[0], 'max'),
              (group_abbrev + "_min_" + statcols[0], 'min'),
              (group_abbrev + "_25perc_" + statcols[0], lambda a: np.percentile(a, 25)),
              (group_abbrev + "_75perc_" + statcols[0], lambda a: np.percentile(a, 75))]) \
        .reset_index()

    for statcol in statcols[1:]:
        if np.percentile(df[statcol], 25) == np.percentile(df[statcol], 75):
            inflated_value = np.percentile(df[statcol], 25)
            print(f"{statcol} is inflated at the value {inflated_value} (comprises >= 50% of the data).")
            print("Only checking outliers for non-inflated values.")
            inflated_values.append(inflated_value)
        else:
            inflated_values.append(np.nan)

        df_aggdata_ = df.groupby(groupcols)[statcol] \
            .agg([(group_abbrev + "_mean_" + statcol, 'mean'),
                  (group_abbrev + "_median_" + statcol, 'median'),
                  (group_abbrev + "_max_" + statcol, 'max'),
                  (group_abbrev + "_min_" + statcol, 'min'),
                  (group_abbrev + "_25perc_" + statcol, lambda a: np.percentile(a, 25)),
                  (group_abbrev + "_75perc_" + statcol, lambda a: np.percentile(a, 75))]) \
            .reset_index()

        df_aggdata = pd.merge(df_aggdata, df_aggdata_, on=groupcols)

    df_outliers = pd.merge(df[statcols + groupcols], df_aggdata, on=groupcols, how='left')

    df_outliers.index = df.index

    for statcol, outlier_factor, inflated_value in zip(statcols, outlier_factors, inflated_values):
        upper = df_outliers[f'{group_abbrev}_75perc_{statcol}']
        lower = df_outliers[f'{group_abbrev}_25perc_{statcol}']
        iqr = upper - lower

        df[f'{statcol}_outlier'] = (df_outliers[statcol] != inflated_value) & \
                                   ((df_outliers[statcol] > upper + outlier_factor * iqr) |
                                    (df_outliers[statcol] < lower - outlier_factor * iqr))

    if keep_agg:
        df = pd.concat((df, df_outliers[[col for col in df_outliers.columns if col not in statcols + groupcols]]), axis=1)

    if not inplace:
        return df


def _N_bma(alpha, beta, l, x):
    '''
    Get sample size for a dataset (x), alpha, beta, and effect size (l). This is one iteration.
    
    Parameters
    ----------
    alpha : float
        The preferred minimum test confidence level (Type I error)
    beta : float
        Test power level (1 - Type II error)
    l : float
        Test size or the minimum meaningful difference to be tested between two samples.
    x : np.array
        The values used to get statistics from the population to calculate sample size

    Returns
    -------
    (float) The sample size estimation
    '''
    z_alpha2 = stats.norm.ppf(alpha / 2)
    z_beta = stats.norm.ppf(beta)
    sigma = x.std()

    nh = ((z_alpha2 + z_beta) ** 2 * sigma ** 2) / l ** 2
    return nh


def sample_size_bma(x_sample, alpha=0.1, beta=0.15, l=0.05, iters=1000):
    '''
    Bootstrap Sample Size calculation as described in "Sample Size Calculations in Clinical Research 2nd ed", by
    S. Chow, et al., (Chapman and Hall, 2008) WW. on page 350 (section 13.3).

    Parameters
    ----------
    x_sample : array-like
        The values used to get statistics from the population to calculate sample size
    alpha : float
        The preferred minimum test confidence level (Type I error)
    beta : float
        Test power level (1 - Type II error)
    l : float
        Test size or the minimum meaningful difference to be tested between two samples.
    iters : int
        Number of iterations of calculating a sample size within the bootstrap sample sizse calculation scheme. (Most cases: 1000)

    Returns
    -------
    (float) The estimated minimum sample size for the parameters given
    '''
    nh_all = []
    x_sample = np.array(x_sample)

    for h in range(iters):
        x = np.random.choice(x_sample, x_sample.shape)
        nh_all.append(_N_bma(alpha, beta, l, x))

    N_bma = np.ceil(np.median(nh_all))  # Bootstrap-Median Approach
    return int(np.ceil(N_bma))


def _multi_fishers_exact(work):
    '''
    Run Fisher's Exact Test for one pair of testing groups.
    '''
    _within, testing_groups, df_contingencies, target_a, target_b = work

    print(f"[{current_process().pid}] Testing {testing_groups} within {_within} ...")

    # Fisher's Exact Test uses odds ratios, so we capture this contingency table
    contingency_table = df_contingencies[list(testing_groups)].loc[[target_a, target_b], :]
    odds_ratio, p_value = stats.fisher_exact(contingency_table)

    # Keep conversion rates for presentation
    proportions_a = df_contingencies[list(testing_groups)].loc[target_a, :] / \
                    df_contingencies[list(testing_groups)].loc['total', :]

    proportions_b = df_contingencies[list(testing_groups)].loc[target_b, :] / \
                    df_contingencies[list(testing_groups)].loc['total', :]

    df_results = {'within': _within,
                  'testing_group_1': testing_groups[0],
                  f'{target_a.lower().replace(" ", "")}_1': df_contingencies[list(testing_groups)[0]][target_a],
                  f'{target_b.lower().replace(" ", "")}_1': df_contingencies[list(testing_groups)[0]][target_b],
                  f'totals_1': df_contingencies[list(testing_groups)[0]]['total'],
                  f'proportion_1_{target_a.lower().replace(" ", "")}': proportions_a[testing_groups[0]],
                  f'proportion_1_{target_b.lower().replace(" ", "")}': proportions_b[testing_groups[0]],
                  'testing_group_2': testing_groups[1],
                  f'{target_a.lower().replace(" ", "")}_2': df_contingencies[list(testing_groups)[1]][target_a],
                  f'{target_b.lower().replace(" ", "")}_2': df_contingencies[list(testing_groups)[1]][target_b],
                  f'totals_2': df_contingencies[list(testing_groups)[1]]['total'],
                  f'proportion_2_{target_a.lower().replace(" ", "")}': proportions_a[testing_groups[1]],
                  f'proportion_2_{target_b.lower().replace(" ", "")}': proportions_b[testing_groups[1]],
                  'odds_ratio': odds_ratio,
                  'p_value': p_value,
                  f'top_group_{target_a.lower().replace(" ", "")}': proportions_a.sort_values(ascending=False).index[0],
                  f'top_group_{target_b.lower().replace(" ", "")}': proportions_b.sort_values(ascending=False).index[0]}

    return df_results


def multi_fishers_exact(df_testing, test_group_col, target_a, target_b, within='all', n_jobs=12):
    '''
    Run pair-wise Fisher's exact tests between all test groups within each (or the) subset in the data. The data in the columns in question do not
    need to be aggregated, as this function will run the group bys.

    The Fisher's Exact Test is only applicable between two test groups of values, where the variance of each group is roughly equal.
    Breaking the data up into subsets before dividing out test groups may be necessary if this assumption is only met within subsets. This function
    allows you to break up the data into subsets (by a column) and then test groups to run pair-wise Fisher's Exact Tests.

    So, e.g., suppose you're working to test if the kids on the left side of the class tend to have a higher test score than the kids on the right
    side of the class. You may have a data set like this:

        student_name | week | side_of_class | correct_answers | incorrect_answers
        'billy'         1         'left'             20            80
        'billy'         2         'right'            20            80
         'joe'          2         'right'            40            60

                                        ...

    (I.e., every week we take a quiz, no assigned seats, sometimes students sit on the left and sometimes on the right).

    Parameters
    ----------
    df_testing : pd.DataFrame
        The data with all the test groups and subsets. All columns needed for this function must be in the dataframe.
        * In the example above, the data set here would just be that data set.
    test_group_col : str
        The column containing the test groups. Within any subset (from `within`), you could have multiple test groups which you'd like to compare
        between one another. This is the column containing the values delineating which test group each row belongs to.
        * In the example, the test_group_col would be `side_of_class`.
    target_a : str
        The column containing the number of observations of Target A (maybe your main target of interest). There should be some concept of "total"
        when you add up `target_a` and `target_b`.
        * I'd probably pick `correct_answers` as `target_a`
    target_b : str
        The complement to `target_a`.
        * I'd probably pick `incorrect_answers` as `target_b`
    within : str
         The subsets to divide the data such that within any subset, comparing test groups is valid.
         * We COULD assume that the variance of correct_answers is roughly the same among all students in the class. Or, we could say that comparing
           scores between class sides only makes sense for a particular student. In the latter case, you'd choose `student_name` as `within`. In the
           former case, you'd just use `within='all'`.
    n_jobs : int
        The number of processes to use when running the Fisher's Exact Tests.

    Returns
    -------
    (pd.DataFrame) An output containing all the results from the tests including p-values, odds ratios, and aggregate data.
    '''
    df_testing['total'] = df_testing[target_a] + df_testing[target_b]

    if within == 'all':
        within_samples = [within]
        df_contingencies_all = {within: df_testing.groupby(test_group_col)[[target_a, target_b, 'total']].sum().T}
    else:
        # There needs to be at least two test groups to test on, so we only get samples `within` that contain at least two test groups
        vcounts = df_testing.groupby(within)[test_group_col].nunique()
        within_samples = vcounts[vcounts > 1].index.tolist()
        df_testing = df_testing[df_testing[within].isin(within_samples)]
        df_contingencies_all = df_testing.groupby([within, test_group_col])[[target_a, target_b, 'total']].sum().T

    allwork = []

    for _within in within_samples:
        df_contingencies = df_contingencies_all[_within]
        testing_groups_all = list(combinations(df_contingencies.columns, 2))
        for testing_groups in testing_groups_all:
            allwork.append((_within, testing_groups, df_contingencies, target_a, target_b))

    with Pool(n_jobs) as p:
        results = p.map(_multi_fishers_exact, allwork)

    return pd.DataFrame(results)


def smooth(y, window):
    '''
    Use convolution smoothing to reduce oscillations in an array, `y`, within a window of size `window`

    Parameters
    ----------
    y : list-like
        The array
    window : int
        Size of window for convolutions

    Returns
    -------
    (np.array) Smoothed array, of the same size as y.
    '''

    y = np.array(y)
    window = np.int(window)
    box = np.ones(window)/window
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth