from uszipcode import SearchEngine
import pandas as pd


def merge_zips(df, zip_column, simple=True):
    zip_search = SearchEngine(simple_zipcode=simple)

    if zip_column == 'zipcode':
        df.rename(columns={'zipcode': 'zipcode_orig'},
                  inplace=True)
        zip_column = 'zipcode_orig'

    zips = df[zip_column].unique()
    df_zips = pd.DataFrame([{**zip_search.by_zipcode(z).to_dict(),
                             **{zip_column: z}} for z in zips])

    df = df.merge(df_zips,
                  on=zip_column,
                  how='left')

    return df