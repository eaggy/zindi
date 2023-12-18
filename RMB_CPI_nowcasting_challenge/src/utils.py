# -*- coding: utf-8 -*-
"""
The file contains python code useful functions.

Created on 26.09.2023

@author: ihar
"""

import os
from typing import Iterable
import numpy as np
import pandas as pd
from pmdarima.arima import ndiffs, nsdiffs
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_percentage_error


def calculate_metric(y_test: Iterable,
                     y_pred: Iterable,
                     metric: str) -> float:
    """Calculate error between predicted and test data.

    Args:
        y_test: Test Y data.
        y_pred: Predicted Y data.
        metric: Metric name to calculate error.

    Returns:
        Calculated error.

    """
    error = None
    if metric == "mae":
        error = mean_absolute_error(y_test, y_pred)
    elif metric == "mape":
        error = mean_absolute_percentage_error(y_test, y_pred)
    elif metric == "mse":
        error = mean_squared_error(y_test, y_pred, squared=True)
    elif metric == "rmse":
        error = mean_squared_error(y_test, y_pred, squared=False)
    elif metric == "r2":
        error = r2_score(y_test, y_pred)
    return error


def extend_index(df: pd.DataFrame, periods: int) -> pd.Index:
    """Extend index of time sorted DataFrame by adding points after maximal date.

    Args:
        df: dataFrame to extend index.
        periods: Numbers of points used to extend index.

    Returns:
        Extended index of DataFrame.

    """
    idx_name = df.index.name
    index = df.index.union(
        pd.date_range(df.index[-1] +
                      df.index.freq,
                      periods=periods,
                      freq=df.index.freq)
    )
    index.name = idx_name
    return index


def generate_file_path(path_in: str, end_month: str) -> str:
    """Generate file path using initial path and the last month.

    Args:
        path_in: Initial path.
        end_month: Last month in format "%Y-%m".

    Returns:
        Generated file path.

    """
    file_name = os.path.normpath(path_in.split("/")[-1])
    file_name, file_extension = os.path.splitext(file_name)
    file_name = f"{file_name}_{end_month}"
    file_name = file_name + file_extension
    path_split = path_in.split("/")[:-1]
    path_split.append(file_name)
    file_path = "/".join(path_split)
    return file_path


def get_d_sd(y: pd.Series) -> tuple[int, int]:
    """Conduct ADF, KPSS, and PP tests of stationarity, and
       CH and OCSB tests of seasonality. Then select maximal values of
       differencing and seasonal differencing from each group of tests.

    Args:
        y: Time series to conduct tests.

    Returns:
        Maximal values of differencing and seasonal differencing.

    """
    try:
        adf_diffs = ndiffs(y, alpha=0.05, test="adf", max_d=6)
    except:
        adf_diffs = 0
    try:
        kpss_diffs = ndiffs(y, alpha=0.05, test="kpss", max_d=6)
    except:
        kpss_diffs = 0
    try:
        pp_diffs = ndiffs(y, alpha=0.05, test="pp", max_d=6)
    except:
        pp_diffs = 0
    try:
        ch_sdiffs = nsdiffs(y, m=12, max_D=6, test="ch")
    except:
        ch_sdiffs = 0
    try:
        ocsb_sdiffs = nsdiffs(y, m=12, max_D=6, test="ocsb")
    except:
        ocsb_sdiffs = 0
    n_diffs = max(adf_diffs, kpss_diffs, pp_diffs)
    n_sdiffs = max(ch_sdiffs, ocsb_sdiffs)
    return n_diffs, n_sdiffs


def set_seed(seed: int) -> np.random.RandomState:
    """Set seed for code that uses the singleton RandomState
       to makes the random numbers predictable.

    Args:
        seed: Seed to initialize random generator.

    Returns:
        Random state.

    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    return np.random.RandomState(seed)
