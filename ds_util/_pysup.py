import os
import sys
import re
import string
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from collections import Counter
from itertools import chain, combinations
from copy import deepcopy
from glob import glob

try:
    # In case we're in the tensorflow environment
    import lightgbm as lgb
except:
    pass

from dstools.ds_statsml._alt_metrics import *
from time import time


def blockPrinting(func):
    """
    *Decorator* to block printing of anything within the defined function. Handy for defining functions where you know several sub-functions print
    a whole lot of junk.

    For example, if you have:

        def printfunc():
            print('blah')

        @blockPrinting
        def outerfunc():
            printfunc()
            print('outer blah')

        >>> outerfunc()
        [Out 1]:

    You get nothing.
    """
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        with open(os.devnull, "w") as devNull:
            original = sys.stdout
            sys.stdout = devNull    # suppress printing
            func(*args, **kwargs)
            sys.stdout = original   # re-enable printing

    return func_wrapper


def depth(d, level=1):
    """
    Get the maximum depth of a dictionary.

    Parameters
    ----------
    d : dict
        Dictionary to determine depth
    level : recursive variable
        LEAVE EQUAL TO 1 (recursive function)

    Returns
    -------
    (int) The maximum depth of the dictionary (i.e., the most number of nested dictionaries).
    """
    if not isinstance(d, dict) or not d:
        return level
    return max(depth(d[k], level + 1) for k in d)


def flatten_list(l):
    '''
    Convert a list of lists (two-dimensional shape) to a single list.

    Parameters
    ----------
    l : list-like
        The original list

    Returns
    -------
    (list) A flattened version of the original list of lists.
    '''
    return [item for sublist in l for item in sublist]


def timeit(method):
    """
    *Decorator* function to log the time to execute the function defined each time it's called.
    """
    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print(f'{method.__name__}  {round((te - ts) * 1000, 3)} ms')
        return result
    return timed


def get_array_batches(a, max_batch_size=4):
    """
    Given an array, break it up into batches of `max_batch_size` or less. In general, the batches should be of size `max_batch_size`. But,
    if this number doesn't divide evenly into `len(a)`, then the batches that are not of size `max_batch_size` will be `max_batch_size - 1`.

    Parameters
    ----------
    a : array-like
        The array to be divided into batches.
    max_batch_size : int, optional
        The maximum size for each batch

    Returns
    -------
    (list of lists) A list containing batches (each in the form of a list).
    """
    a = np.array(a)
    cv = KFold(n_splits=int(np.ceil(len(a) / max_batch_size)))

    return [list(a[x[1]]) for x in cv.split(a)]


def numerate_dupes(x):
    e_counts = dict(Counter(x))
    ooms = {e: int(np.floor(np.log10(e_counts[e]))) for e in x}
    x_num = deepcopy(x)
    duplicates = {k: 0 for k in x}

    for i, e in enumerate(x):
        if e_counts[e] > 1:
            duplicates[e] += 1
            x_num[i] = e + str(duplicates[e]).zfill(ooms[e] + 1)

    return x_num


def make_column_names(strings):
    remove_punct = str.maketrans('', '', string.punctuation.replace('_', ''))
    cols = [x.lower().translate(remove_punct).replace(' ', '_') for x in strings]
    cols = numerate_dupes(cols)
    return cols


def glob_re(filepattern, dirpath, fullpath=True):
    if fullpath:
        filepaths = [f for f in glob(dirpath + "/**", recursive=True) if not os.path.isdir(f)]

        def filter_func(f):
            return re.compile(filepattern).match(f[max(f.rfind('/'), f.rfind('\\')) + 1:])

        return list(filter(filter_func, filepaths))

    else:
        return list(filter(re.compile(filepattern).match, os.listdir(dirpath)))


def hist_data(a, n_bins, kws=None):
    if kws is None:
        kws = {}

    hist_data = np.histogram(a, bins=n_bins, **kws)
    hist_data = pd.DataFrame(data=zip(hist_data[1], hist_data[0]),
                             columns=['bin_edges', 'values'])

    return hist_data


def powerset(items):
    '''
    Given an iterable of items, return a generator of all possible combinations (in tuples).

    For single combinations (say, groups of m), just use itertools.combinations(items, m)

    Parameters
    ----------
    items : iterable
        The items you'd like to get combinations for

    Returns
    -------
    (iterable) Generator of all possible combinations in tuples
    '''
    s = list(items)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def powerset_dataframe(items, index_name='powerset'):
    '''
    Given an iterable of items, return a Pandas DataFrame with the items as columns, and True or False for values.
    Add a column (named `index_name`) indexing the powersets. So, row i (`index_name` == i), column m is True if
    column m is in powerset i.

    Parameters
    ----------
    items : iterable
        The items to get combinations for
    index_name : str
        The name of the column indexing combinations

    Returns
    -------
    (pandas.DataFrame) The dataframe containing combinations

    '''
    item_combinations = []
    for group in powerset(items):
        item_combinations.append({g: True for g in group})

    df_powersets = pd.DataFrame(item_combinations)
    df_powersets.fillna(False, inplace=True)
    df_powersets.reset_index(inplace=True)
    df_powersets.rename(columns={'index': index_name}, inplace=True)

    return df_powersets


def cumulative_count_new(x):
    items = set()
    cum_counts = []

    for x_ in x:
        items.update(x_)
        cum_counts.append(len(items))

    return cum_counts


def make_append_file(filepath=None, first_line="", add_line='', break_lines=True):
    br = '\n' if break_lines else ''

    if filepath is not None and not os.path.isfile(filepath):
        with open(filepath, "w") as f:
            f.write(first_line)

    with open(filepath, "a") as f:
        f.write(br + add_line)


def priorities_to_weights(priorities):
    scaler = MinMaxScaler()
    priorities = np.array(priorities)
    priorities = np.reshape(priorities, (-1, 1))
    weights = 2 - scaler.fit_transform(priorities, (-1, 1))
    weights = weights / weights.sum()
    return weights


def get_jupyter_vars(include_ipython=False):
    if include_ipython:
        ipython_vars = []
    else:
        # These are the usual ipython objects, including this one you are creating
        ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

    # Get a sorted list of the objects and their sizes
    variables = {x: sys.getsizeof(globals().get(x)) for x in dir()
                 if not x.startswith('_')
                 and x not in sys.modules
                 and x not in ipython_vars}

    return pd.Series(variables)