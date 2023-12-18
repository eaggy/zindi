# -*- coding: utf-8 -*-
"""
The file contains grids definition for hyperparameters tuning.

Created on 26.09.2023

@author: ihar
"""

from functools import partial
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import r_regression, f_regression
from sklearn.feature_selection import mutual_info_regression
from src.utils import set_seed
from settings import SEED

rng = set_seed(SEED)
_mutual_info_regression = partial(mutual_info_regression, random_state=42)


lasso_alphas = [0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007,
                0.00008, 0.00009, 0.0001, 0.00015, 0.0002, 0.00025, 0.0003,
                0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]

ridge_alphas = [0.1, 0.15, 0.2, 0.5, 1., 1.5, 2., 5., 10.,
                15., 20., 50., 100., 150., 200., 300., 500.]

grid = {

    "ab": {
        "estimator__max_depth": [1, 2],
        "estimator__min_samples_leaf": [1, 2],
        "estimator__max_features": [0.7, 0.8, 0.9, 1.],
        "learning_rate": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2,
                          0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
    },

    "br": {
        "estimator__max_depth": [1, 2],
        "estimator__min_samples_leaf": [5, 6],
        "max_features": [0.05, 0.1, 0.15, 0.2, 0.25]
    },

    "cb": {
        "learning_rate": [0.03, 0.04],
    },

    "knn": {
        "n_neighbors": range(3, 41),
        "p": [1, 2]
    },

    "lars": {
        "n_nonzero_coefs": [0.035, 0.045, 0.05],
    },

    "lasso": {"alpha": lasso_alphas},

    "lgbm": [
        {
            "boosting_type": ["gbdt"],
            "n_estimators": [50, 100, 150, 200],
            "max_depth": [1, 2],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_samples": [20, 25, 30],
        },
        {
            "boosting_type": ["dart"],
            "n_estimators": [250],
            "max_depth": [3, 5],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_samples": [10, 15],
        }
    ],

    "rf": {
        "criterion": ["squared_error", "absolute_error"],
        "max_depth": [1, 2],
        "min_samples_leaf": [5, 6]
    },

    "ridge": {"alpha": ridge_alphas},

    "sarimax": {
        "p": [0, 1, 2, 3],
        "q": [0, 1, 2, 3],
        "P": [0, 1, 2],
        "Q": [0, 1, 2],
        "trend": ["n"]
    },

    "sgd": {
        "l1_ratio": [0., 0.1, 0.25],
        "alpha": [0.001, 0.002, 0.005],
        "eta0": [0.005, 0.01, 0.02, 0.03, 0.04],
    },

    "svr": {
        "kernel": ["poly", "rbf", "sigmoid"],
        "gamma": ["scale", "auto"],
        "C": [0.0001, 0.001, 0.01, 0.1],
        "epsilon": [0.001, 0.002, 0.003],
    },

    "xgb": {
        "max_depth": [1, 2],
        "colsample_bytree": [0.4, 0.6],
        "gamma": [0.00005, 0.0001, 0.0002, 0.0005],
    },

    # pipe models
    "ridge_kb": {
        "selector": [
            SelectKBest()
        ],
        "selector__k": [0.1, 0.25, 0.5, 0.75, 0.9],
        "selector__score_func": [f_regression,
                                 r_regression,
                                 _mutual_info_regression],
        "ridge__alpha": [0.1, 1., 10., 100.],
    },

    "svr_kb": {
        "selector": [
            SelectKBest()
        ],
        "selector__k": [0.25, 0.5, 0.75],
        "selector__score_func": [f_regression],
        "svr__kernel": ["poly", "rbf"],
        "svr__gamma": ["scale", "auto"],
        "svr__C": [0.001, 0.01, 0.1, 1],
        "svr__epsilon": [0.001]
    },

}
