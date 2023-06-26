import numpy as np
from sklearn.metrics import mean_squared_error


def APE(actual, forecast):
    return np.abs((actual - forecast) / (actual))


def APE_inv(actual, forecast):
    return np.abs(actual / (actual - forecast))


def sAPE(actual, forecast):
    return np.abs(actual - forecast) / ((actual + forecast) / 2)


def sAPE2(actual, forecast):
    return np.abs((actual - forecast) * ((actual + forecast) / 2) ** 0.5)


def AAPE(actual, forecast):
    return np.arctan(APE(actual, forecast))


def AAPE_inv(actual, forecast):
    return np.arctan(APE_inv(actual, forecast))


def MAPE(actual, forecast):
    return np.mean(APE(actual, forecast))


def MdAPE(actual, forecast):
    return np.median(APE(actual, forecast))


def sMAPE(actual, forecast):
    return np.mean(sAPE(actual, forecast))


def sMAPE2(actual, forecast):
    return np.mean(sAPE2(actual, forecast))


def MAAPE(actual, forecast):
    return np.mean(AAPE(actual, forecast))


def MAAPE_inv(actual, forecast):
    return np.mean(AAPE_inv(actual, forecast))


def RMSE(actual, forecast):
    return np.sqrt(mean_squared_error(actual, forecast))


def median_within(a):
    '''
    Adjusted median to keep the value in the array
    :param a:
    :return:
    '''
    a_ = np.array(a).copy()
    a_.sort()

    return a_[len(a_) // 2]