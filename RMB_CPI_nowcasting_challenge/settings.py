# -*- coding: utf-8 -*-
"""
The file contains all important settings.

Created on 26.09.2023

@author: ihar
"""

# proxy settings #
##################
USE_PROXY = False
PROXY_USER = ""
PROXY_PASSWORD = ""
PROXY_URL = ""
PROXY_PORT = ""

# model settings #
##################
END_MONTH = "2023-10"
START_MONTH = "2008-01"
# transformations of y-data (CPI)
TRANSFORMATIONS = [("log", None), ("diff", 1)]
# lags of y-data (CPI) to add in regressors (x-data)
Y_LAGS = [1, 2, 3, 6, 12, 18, 24, 36]
ADD_DUMMY = True
ADD_COVID = True
# use last two years (24 months) for models validation
VALIDATION_STEPS = 24
# forecast next month
FORECAST_STEPS = 1
EVALUATION_METRIC = "rmse"
MODELS = ["ab", "br", "cb", "knn", "lars", "lasso", "lgbm", "rf", "ridge",
          "sarimax", "sgd", "svr", "xgb"]
# some models are used with  select k-best features selection ("kb")
FEATURE_SELECTION = {
    "ab": None,
    "br": None,
    "cb": None,
    "knn": None,
    "lars": None,
    "lasso": None,
    "lgbm": None,
    "rf": None,
    "ridge": "kb",
    "sarimax": None,
    "sgd": None,
    "svr": "kb",
    "xgb": None
}
# information criterion is used to find optimal SARIMA order
# possible values are "aic", "bic", and "hqic"
IC = "hqic"
# range of acceptable accuracy of best model to select other good models
# for creation "best_max" model
RANGE = 0.1
# random seed
SEED = 42

# URLs #
########
URL_STATSSA_TS = "https://www.statssa.gov.za/timeseriesdata/Ascii/P0141%20-%20CPI(COICOP)%20from%20Jan%202008%20({}).zip"
URL_API = "https://custom.resbank.co.za/SarbWebApi/WebIndicators/Shared/GetTimeseriesObservations/"
URL_DOWNLOAD_FACILITY = "https://www.resbank.co.za/bin/sarb/custom/downloadfacility/"

# paths #
#########
STATSSA_FILE_LOCATION = "./data/statssa.csv"
SARB_FILE_LOCATION = "./data/sarb.csv"
OPTIMAL_COLUMNS_LOCATION = "./data/optimal_columns/"
OUTPUT_LOCATION = "./data/output/"

MAPPING = {
            "CPS00000": "headline CPI",
            "CPS01000": "food and non-alcoholic beverages",
            "CPS02000": "alcoholic beverages and tobacco",
            "CPS03000": "clothing and footwear",
            "CPS04000": "housing and utilities",
            "CPS05000": "household contents and services",
            "CPS06000": "health",
            "CPS07000": "transport",
            "CPS08000": "communication",
            "CPS09000": "recreation and culture",
            "CPS10000": "education",
            "CPS11000": "restaurants and hotels",
            "CPS12000": "miscellaneous goods and services"
        }
