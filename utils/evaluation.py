# utils/evaluation.py

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def calculate_nse(y_true, y_pred):
    """
    Nash-Sutcliffe Efficiency (NSE)
    """
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator) if denominator != 0 else np.nan


def calculate_pbias(y_true, y_pred):
    """
    Percent Bias (PBIAS)
    """
    bias = np.sum(y_true - y_pred)
    total = np.sum(y_true)
    return 100 * bias / total if total != 0 else np.nan


def calculate_metrics(y_true, y_pred):
    """
    Dictionary of common evaluation metrics
    """
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "nse": calculate_nse(y_true, y_pred),
        "pbias": calculate_pbias(y_true, y_pred),
    }
