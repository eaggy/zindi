# -*- coding: utf-8 -*-
"""
The file contains python code to perform CPI prediction.

Created on 26.09.2023

@author: ihar
"""

from timeit import default_timer as timer
from datetime import timedelta
from src.model import Model
from settings import START_MONTH, END_MONTH, TRANSFORMATIONS
from settings import Y_LAGS, ADD_DUMMY, ADD_COVID
from settings import EVALUATION_METRIC, VALIDATION_STEPS, FORECAST_STEPS
from settings import MODELS


if __name__ == "__main__":
    model = Model(start_month=START_MONTH,
                  end_month=END_MONTH,
                  transformations=TRANSFORMATIONS,
                  y_lags=Y_LAGS,
                  add_dummy=ADD_DUMMY,
                  add_covid=ADD_COVID,
                  evaluation_models=MODELS,
                  evaluation_metric=EVALUATION_METRIC,
                  validation_steps=VALIDATION_STEPS,
                  forecast_steps=FORECAST_STEPS)
    start_time = timer()
    model.evaluate_forecast()
    end_time = timer()
    print(timedelta(seconds=end_time - start_time))
