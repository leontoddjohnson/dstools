import numpy as np
import pandas as pd


def get_iplot_text(data, names=None, round_digits=2):

    data_ = data.copy()
    
    if names is None:
        if hasattr(data_, 'columns'):
            names = data_.columns.tolist()
            num_cols = [col for col in names if data_[col].dropna().shape[0] > 0 and
                        isinstance(data_[col].dropna().iloc[0], int) or isinstance(data_[col].dropna().iloc[0], float)]
            data_[num_cols] = data_[num_cols].round(round_digits)
            data_ = data_.values

        else:
            names = [f'var{i + 1}' for i in range(data_.shape[1])]
            data_ = pd.DataFrame(columns=names, data=data_)
            num_cols = [col for col in names if data_[col].dropna().shape[0] > 0 and
                        isinstance(data_[col].dropna().iloc[0], int) or isinstance(data_[col].dropna().iloc[0], float)]
            data_[num_cols] = data_[num_cols].round(round_digits)
            data_ = data_.values

    text_array = np.char.array([names[0]] * data_.shape[0])
    text_array = text_array + ": " + np.char.array(data_[:, 0], unicode=True) + "<br>"

    for i in range(1, len(names)):
        text_array_ = np.char.array([names[i]] * data_.shape[0])
        text_array = text_array + text_array_ + ": " + np.char.array(data_[:, i], unicode=True) + "<br>"

    return text_array