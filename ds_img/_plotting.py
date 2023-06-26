import matplotlib.pyplot as plt
import skimage
from ._processing import *
from matplotlib.patches import Rectangle


def show_image(img, gray=False):
    '''
    Show an image in matplotlib fashion given some array image.

    Examples:
    image_io = skimage.io.imread(FILEPATH)
    fig, axis = show_image(image_io)

    Parameters
    ----------
    img : str
        The name of a file or full url pointing to the iamge you'd like to plot.
    gray : bool (Default: False)
        Whether to plot the image in grayscale or not.

    Returns
    -------
    *tuple*: (plt.Figure, plt.Axes.axis)
    '''
    if not hasattr(img, 'shape'):
        raise ValueError("The `img` here must be an array. Use either skimage.io.imread, url_to_image, or file_to_image.")

    fig, axis = plt.subplots(1)

    if gray:
        axis.imshow(img, cmap='gray')
    else:
        axis.imshow(img)

    plt.grid(None);
    return fig, axis


def multiplot_img(files=None, urls=None, url_prefix='https://images-na.ssl-images-amazon.com/images/I/',
                  n_cols=3, figsize=(22, 22), titles=None, fontsize=18):
    '''
    Plot multiple images in one frame (many axes, one figure).

    Parameters
    ----------
    files : list
        The *filepaths* to the images (.png, .jpg, etc. files)
    urls : list
        The list of urls (or url suffixes/filenames) to the images. If these do not have 'http' in the strings, then the function will use the
        `url_prefix` as the prefix for the url.
    n_cols : int
        The number of columns to plot.
    figsize : tuple
        Figure size
    titles : list (same length as `files` or `urls`)
        These are the titles corresponding to each image in `files` or `urls`
    fontsize : int
        Font sze for each axis in the figure

    Returns
    -------
    (matplotlib.pyplot.Figure) The figure with the plots.

    '''
    if urls is not None:
        images = urls
    elif files is not None:
        images = files
    else:
        raise ValueError("You need to have files or urls of images to plot ...")

    fig = plt.figure(figsize=figsize)
    n_rows = int(np.ceil(len(images) / n_cols))

    for i, image in enumerate(images):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        if urls is not None:
            ax.imshow(url_to_image(image, url_prefix))
        else:
            ax.imshow(file_to_image(image))

        if titles is not None:
            ax.set_title(titles[i], fontsize=fontsize)
        plt.grid(None)

    fig.tight_layout()

    return fig


def plot_rectangle(axis, lower_left_point, width, height, ls='--', lw=3, color='red', fill=False, label=None):
    '''
    Plot a rectangle over an axis.

    Parameters
    ----------
    axis : matplotlib axis
        The axis to plot the rectangle on.
    lower_left_point : tuple
        A tuple containing the coordinates for the *lower left* point of the rectangle you'd like to plot.
    width : float
        See matplotlib.patches.Rectangle
    height : float
        See matplotlib.patches.Rectangle
    ls : str
        See matplotlib.patches.Rectangle
    lw : int
        See matplotlib.patches.Rectangle
    color : str
        See matplotlib.patches.Rectangle
    fill : bool
        See matplotlib.patches.Rectangle
    label : str, or None
        See matplotlib.patches.Rectangle

    Returns
    -------
    *Nothing*

    Just plot the rectangle on the axis provided.
    '''
    x = lower_left_point[0]
    y = lower_left_point[1]

    p = Rectangle((x, y), width=width, height=height, ls=ls, lw=lw, color=color, fill=fill, label=label)
    axis.add_patch(p)


def show_imagely(image_id, df_labels, colors=('red', 'orange', 'blue', 'pink', 'yellow'), show_3grid=False):
    '''
    Plot labeled images from Supervise.ly along with rectangles containing objects.

    Parameters
    ----------
    image_id : str
        Image ID corresponding to an 'image_id' from the Supervise.ly output
    df_labels : pd.DataFrame
        The dataframe output from `supervisely_to_df`
    colors : list of strings
        The colors (in order of items in df_labels['object_titles']) that you'd like to use in plotting the objects in the image with `image_id`.
    show_3grid : bool
        Show the 3 x 3 grid of lines over the image

    Returns
    -------
    matplotlib.Figure, matplotlib.Axes
    '''
    df_ = df_labels[df_labels['object_titles'].apply(len) > 0]
    items = df_[df_.image_id == image_id]['object_titles'].iloc[0]
    colors = list(colors)[:len(items)]

    image_io = skimage.io.imread(f'./images/{image_id}')
    fig, axis = show_image(image_io)
    fig.set_size_inches(12, 12)

    for item, color in zip(items, colors):
        x_centroid = df_labels[df_labels['image_id'] == image_id][f'{item}|object_x_centroid'].iloc[0]
        y_centroid = df_labels[df_labels['image_id'] == image_id][f'{item}|object_y_centroid'].iloc[0]
        width = df_labels[df_labels['image_id'] == image_id][f'{item}|avg_object_width'].iloc[0]
        height = df_labels[df_labels['image_id'] == image_id][f'{item}|avg_object_height'].iloc[0]

        left_x = x_centroid - width / 2
        lower_y = y_centroid - height / 2

        axis.scatter([x_centroid], [y_centroid], color=color, s=50, label=f'{item} Center')
        plot_rectangle(axis, (left_x, lower_y), width, height, color=color, label=f'Average {item}')

    if show_3grid:
        w = df_[df_.image_id == image_id]['image_width'].iloc[0]
        h = df_[df_.image_id == image_id]['image_height'].iloc[0]

        axis.axvline(1 / 3 * w, color='black', ls='--')
        axis.axvline(2 / 3 * w, color='black', ls='--')
        axis.axhline(1 / 3 * h, color='black', ls='--')
        axis.axhline(2 / 3 * h, color='black', ls='--')

    axis.legend()
    print(image_id)

    return fig, axis
