

def tf_projector(df, embedding_cols, report_cols, save_path=".", name=''):
    '''
    View multidimensional data using Tensorflow's Tensorboard Projector.

    Parameters
    ----------
    df : Pandas.DataFrame
        DataFrame containing all the vector data (to be plot) as well as *at least* one column to report on.
    embedding_cols : list-like
        Columns that represent the vectors that you'd like to visualize.
    report_cols : list-like
        Columns of metadata (or reporting data) that you'd like to be able to show in the form of labels or colors in the projection.
    save_path : str
        Location to save the vector and metadata tsvs. It will attempt to save *all* data within the columns provided.
    name : str
        Any specific name of this set of data (it will be the prefix of the saved files)

    Returns
    -------
    None
    '''

    df[embedding_cols].to_csv(f'{save_path}/{name}_vectors.tsv', sep='\t', index=False, header=False)
    df[report_cols].to_csv(f'{save_path}/{name}_metadata.tsv', sep='\t', index=False)

    url = 'https://projector.tensorflow.org/'

    print(f"{name}_vectors.tsv and {name}_metadata.tsv saved to {save_path}.")
    print(f"You can visualize these vectors by:")
    print(f"1. Go to {url}.")
    print("2. Select the 'Load' button on the left, and select 'Choose File' for both the vectors and the metadata.")