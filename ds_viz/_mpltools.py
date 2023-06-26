import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


def cleanup_xaxis(axis, stepsize, rotation, round_x=None, round_y=None):
    start, end = axis.get_xlim()
    axis.xaxis.set_ticks(np.arange(start, end, stepsize))

    for tick in axis.get_xticklabels():
        tick.set_rotation(rotation)

    if round_x is not None:
        axis.xaxis.set_major_formatter(FormatStrFormatter(f'%.{round_x}f'))

    if round_y is not None:
        axis.yaxis.set_major_formatter(FormatStrFormatter(f'%.{round_y}f'))

    return axis


def multiplot(plotters, params, n_cols=3, figsize=(22, 22), fig_filename=None):
    '''

    Parameters
    ----------
    plotters : list, callable
        If you're running scatterplots, an example of `params` could be:

        params = [{'x': 'variable',
                   'y': col,
                   'data': df[df[col] != 0]} for col in columns]

    Alternatively, you can use this framework:

        fig = plt.figure(figsize=figsize)
        n_rows = int(np.ceil(len(groups) / n_cols))

        for i, group in enumerate(groups):
            axis = fig.add_subplot(n_rows, n_cols, i + 1)
            sns.someplot(df[df[grouping_col] == group][value_col], label=group, ax=axis)
            axis.set_title(...)

        fig.tight_layout()

    params
    n_cols
    figsize
    fig_filename

    Returns
    -------
    fig, axes

    You can adjust settings for these post hoc.
    '''
    if not isinstance(plotters, list):
        plotters = [plotters] * len(params)

    n_rows = int(np.ceil(len(params) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    _x = np.arange(0, n_rows * n_cols).reshape(n_rows, n_cols)

    for i, param_set in enumerate(params):
        x_i = np.where(_x == i)[0][0]
        y_i = np.where(_x == i)[1][0]

        axis = axes[x_i, y_i]

        plotters[i](**param_set, ax=axis)

    fig.tight_layout()

    if fig_filename is not None:
        plt.savefig(fig_filename, orientation='landscape')

    return fig, axes