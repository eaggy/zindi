# -*- coding: utf-8 -*-
"""
This is the main file of predictor.

Created on 26.09.2023

@author: ihar
"""

import calendar
import os.path
from typing import Optional, Union
from datetime import datetime
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import SplineTransformer
from src.loader import Loader
from src.evaluator import Evaluator
from src.utils import calculate_metric, extend_index, set_seed
from settings import MAPPING, SEED, OUTPUT_LOCATION, RANGE
import warnings

warnings.filterwarnings("ignore")


class Model:
    def __init__(self,
                 start_month: str,
                 end_month: str,
                 transformations: list,
                 y_lags: Union[int, list[int]],
                 add_dummy: bool,
                 add_covid: bool,
                 evaluation_models: list[str],
                 evaluation_metric: str,
                 columns: Optional[tuple] = (),
                 validation_steps: Optional[int] = 1,
                 forecast_steps: Optional[int] = 1,
                 ) -> None:
        """
        Args:
            start_month: First month to load data in format "%Y-%m".
            end_month: Last month to load data in format "%Y-%m".
            transformations: List of transformations applied to Y data.
            forecast_steps: Number of forecast steps.
            y_lags: List of lags used to create lags of y column (CPI) and
                    add these columns to X data.
            add_dummy: Add seasonal dummy variables to X data, if True.
            add_covid: Add COVID event to X data, if True.
            evaluation_models: Models names for evaluation and prediction.
            evaluation_metric: Metric used for evaluation.
            columns: CPI names used to create models.
                     All CPI components are used if empty.
            validation_steps: Number of validation steps.
            forecast_steps: Number of forecast steps.

        """
        pd.set_option("display.max_colwidth", 100)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.float_format", "{:.8f}".format)
        set_seed(SEED)
        self.start_month = start_month
        self.end_month = end_month
        self.columns = columns
        self.forecast_steps = forecast_steps
        self.validation_steps = validation_steps
        self.add_dummy = add_dummy
        self.add_covid = add_covid
        self.evaluation_models = evaluation_models
        self.evaluation_metric = evaluation_metric
        self.y_lags = y_lags
        self.x_data, self.y_data = Loader(self.start_month,
                                          self.end_month,
                                          self.forecast_steps).load_all_data()
        self.x_columns = {"X": self.x_data.columns.to_list()}
        if self.columns:
            # forecast only for CPIs defined in `self.columns`
            self.y_data = self.y_data.loc[:, self.columns]
        else:
            # forecast for all CPIs
            self.columns = self.y_data.columns.to_list()
        for column in self.columns:
            self.x_columns[column] = []
        self.transformations = transformations
        self.__y_data_transformations = []
        self.__transform_y_data()
        if self.add_dummy:
            self.__add_dummy_variables("m", True)
        if self.add_covid:
            self.__add_covid("peak")
        self.__add_y_lags()
        self.x_data = self.__remove_nans_rows(self.x_data, "begin")
        self.results = {}

    def __add_covid(self, mode: str) -> None:
        """Add covid event to X data.

         Args:
            mode: "peak" - add peak on April 2020,
                  "step" - add step starting on April 2020,
                  "both" - add peak and step.

        """
        col_name = "COVID"
        ext_index = extend_index(self.y_data, self.forecast_steps)
        covid = pd.DataFrame(index=ext_index)
        if mode in ["peak", "both"]:
            covid[f"{col_name}_peak"] = 0.
            covid.loc[datetime.strptime("2020-04-30", "%Y-%m-%d"),
                      f"{col_name}_peak"] = 1.
        if mode in ["step", "both"]:
            covid[f"{col_name}_step"] = covid.index
            covid[f"{col_name}_step"] = covid[f"{col_name}_step"].apply(
                lambda x:
                1. if x >= datetime.strptime("2020-04-30", "%Y-%m-%d") else 0.)
        if self.x_data is None:
            self.x_data = covid.loc[:, :]
        else:
            self.x_data = pd.concat([self.x_data, covid], axis=1)
        if mode == "both":
            self.x_columns["X"].extend([f"{col_name}_peak",
                                        f"{col_name}_step"])
        elif mode == "peak":
            self.x_columns["X"].append(f"{col_name}_peak")
        elif mode == "step":
            self.x_columns["X"].append(f"{col_name}_step")
        else:
            raise ValueError

    def __add_dummy_variables(self, period: str, use_splines: bool) -> None:
        """Add seasonal dummy variables to X data.

        Args:
            period: Period of seasonality: "m" (monthly) or "q" (quarterly).
            use_splines: Use spline function if True, or peak function if False.

        """
        drop_first = False
        if period == "m":
            period_int = 12
        elif period == "q":
            period_int = 4
        else:
            raise ValueError
        ext_index = extend_index(self.y_data, self.forecast_steps)
        if period == "m":
            data = ext_index.month
        elif period == "q":
            data = ext_index.quarter
        else:
            raise ValueError
        if use_splines:
            transformer = SplineTransformer(
                degree=3,
                n_knots=period_int + 1,
                knots=np.linspace(
                    0, period_int,
                    period_int + 1
                ).reshape(period_int + 1, 1),
                extrapolation="periodic",
                include_bias=True)
            dummy_variables = pd.DataFrame(
                transformer.fit_transform(np.array(data).reshape(-1, 1))
            )
            columns = {i: f"dummy_m_{i+1}" for i in
                       range(dummy_variables.shape[0])}
            dummy_variables.set_index(ext_index, inplace=True)
            dummy_variables.rename(columns=columns, inplace=True)
            if drop_first:
                dummy_variables = dummy_variables.iloc[:, 1:]
        else:
            dummy_variables = pd.get_dummies(data,
                                             prefix=f"dummy_{period}",
                                             drop_first=drop_first,
                                             dtype=int)
            dummy_variables.set_index(ext_index, inplace=True)
        if self.x_data is None:
            self.x_data = dummy_variables.loc[:, :]
        else:
            self.x_data = pd.concat([self.x_data, dummy_variables], axis=1)
        self.x_columns["X"].extend(dummy_variables.columns.to_list())

    def __add_y_lags(self) -> None:
        """Add lagged y column (CPI) to X data.
        Number and order of lags are defined by `self.y_lags`.

        """
        lags_list = self.__generate_lags_list(self.y_lags)
        if lags_list:
            for column in self.columns:
                y = self.y_data.loc[:, column]
                y_lags = self.__create_lags_ser(y, lags_list)
                if self.x_data is None:
                    self.x_data = y_lags.loc[:, :]
                else:
                    self.x_data = pd.concat([self.x_data, y_lags], axis=1)
                self.x_columns[column].extend(y_lags.columns.to_list())
            start_month = datetime.strptime(
                self.start_month, "%Y-%m") + relativedelta(day=31)
            end_month = datetime.strptime(
                self.end_month, "%Y-%m") + relativedelta(
                months=self.forecast_steps,
                day=31
            )
            self.x_data = self.x_data.loc[
                          (self.x_data.index.get_level_values(
                              "date") >= start_month) &
                          (self.x_data.index.get_level_values(
                              "date") <= end_month), :]

    @staticmethod
    def __create_lags_ser(ser: pd.Series, lags: list) -> pd.DataFrame:
        """Create DataFrame with lagged series.

        Args:
            ser: Series to create lagged DataFrame.
            lags: Lags to crate lagged DataFrame.

        Returns:
            DataFrame with lagged series.

        """
        lags_df = None
        for lag in lags:
            shifted_index = ser.index.shift(periods=lag)
            shifted_data = ser.set_axis(shifted_index)
            new_name = f"lag_{lag}_{ser.name}"
            shifted_data.rename(new_name, inplace=True)
            if lags_df is None:
                lags_df = pd.DataFrame(shifted_data)
            else:
                lags_df = pd.concat([lags_df, shifted_data], axis=1)
        return lags_df

    @staticmethod
    def __generate_lags_list(lags: Union[int, list[int]]) -> list[int]:
        """Generate list of lag values between 1 and `lags`, if `lags` is integer.

        Args:
            lags: Maximal lag value or list of lags.

        Returns:
            List of lag values.

        """
        lags_list = []
        if lags:
            if isinstance(lags, int):
                lags_list = list(range(1, lags + 1))
            else:
                lags_list = lags
        return lags_list

    @staticmethod
    def __remove_nans_rows(df: pd.DataFrame, where: str) -> pd.DataFrame:
        """Remove NaNs rows at the beginning of at the end of DataFrame.

        Args:
            df: DataFrame to remove NaNs rows.
            where: Where to delete NaNs rows:
                   "begin" - remove at the beginning of DataFrame,
                   "end" - remove at the end of DataFrame.

        Returns:
            DataFrame without NaNs rows.

        """
        n_nans = []
        if where == "begin":
            position = 0
        elif where == "end":
            position = -1
        else:
            raise ValueError
        for column in df.columns.to_list():
            nans = df.loc[:, column].isnull().astype(int).groupby(
                df.loc[:, column].notnull().astype(int).cumsum()
            ).sum().iloc[position]
            n_nans.append(nans)
        max_n = max(n_nans)
        if max_n > 0:
            if where == "begin":
                df = df.iloc[max_n:, :]
            else:
                df = df.iloc[:-max_n, :]
        return df

    def __revert_y_data(self) -> None:
        """Revert Y data (CPI) by making inverse transformation to get real CPI data.

        """
        for transformation in self.__y_data_transformations[::-1]:
            if transformation[0] == "log":
                self.y_data = self.y_data.apply(np.exp)
            elif transformation[0] == "diff":
                df = pd.concat([transformation[2], self.y_data])
                for i in range(transformation[1], df.shape[0]):
                    df.iloc[i, :] = df.iloc[i, :] + \
                                    df.iloc[i-transformation[1], :]
                self.y_data = df.copy()
            elif transformation[0] == "pct":
                df = pd.concat([transformation[2], self.y_data])
                for i in range(transformation[1], df.shape[0]):
                    df.iloc[i, :] = df.iloc[i, :] * \
                                    df.iloc[i - transformation[1], :] + \
                                    df.iloc[i - transformation[1], :]

                self.y_data = df.copy()
            else:
                raise ValueError

    def __transform_y_data(self) -> None:
        """Transform Y data (CPI).

        """
        for transformation in self.transformations:
            if transformation[0] == "log":
                self.y_data = self.y_data.apply(np.log)
                self.__y_data_transformations.append(transformation)
            elif transformation[0] == "diff":
                transform_list = list(transformation)
                transform_list.append(self.y_data.iloc[0:transformation[1], :])
                self.y_data = self.y_data.diff(periods=transformation[1])
                self.y_data.dropna(inplace=True)
                self.__y_data_transformations.append(tuple(transform_list))
            elif transformation[0] == "pct":
                transform_list = list(transformation)
                transform_list.append(self.y_data.iloc[0:transformation[1], :])
                self.y_data = self.y_data.pct_change(periods=transformation[1])
                self.y_data.dropna(inplace=True)
                self.__y_data_transformations.append(tuple(transform_list))
            else:
                raise ValueError

    def generate_submissions(self, model_name: str) -> None:
        """Generate submission files for model `model_name`.

        Args:
            model_name: Name of model to generate submissions.

        """
        submission = ["ID,Value"]
        submission_data = self.y_data.iloc[-1, :]
        month = calendar.month_name[int(submission_data.name.strftime("%m"))]
        print(submission_data)

        # save monthly data
        for cpi, value in submission_data.iteritems():
            if cpi in MAPPING.keys():
                submission.append(f"{month}_{MAPPING[cpi]},{round(value, 1)}")
        os.makedirs(OUTPUT_LOCATION, exist_ok=True)
        file_name = os.path.join(OUTPUT_LOCATION,
                                 f"submission_{model_name}_{month}.csv")
        with open(file_name, "w", newline="") as handler:
            for line in submission:
                handler.write(f"{line}\r\n")

        # save all data
        file_name_out = os.path.join(OUTPUT_LOCATION,
                                     f"submission_{model_name}_3m.csv")
        submission = ["ID,Value"]
        for month in ["September", "October", "November"]:
            file_name_in = os.path.join(OUTPUT_LOCATION,
                                        f"submission_{model_name}_{month}.csv")
            try:
                with open(file_name_in, "r", encoding="utf-8") as handler_in:
                    for line in handler_in:
                        if line.startswith(month):
                            submission.append(line.strip("\n"))
            except FileNotFoundError:
                for cpi, value in submission_data.iteritems():
                    if cpi in MAPPING.keys():
                        submission.append(
                            f"{month}_{MAPPING[cpi]},0.0")
        with open(file_name_out, "w", newline="") as handler_out:
            for line in submission:
                handler_out.write(f"{line}\r\n")

    def evaluate_best_models(self, column: str) -> tuple[float, pd.Series]:
        """Create best max model for a given CPI, evaluate it, and make prediction.
        The best max model is defined as follows:
           1. All models are evaluated and model with the lowest validation
        metric is selected and named as best model.
           2. Validation threshold is calculated as (1. + RANGE) x the lowest validation
        metric.
           3. All models with validation metrics less than the validation
        threshold are selected.
           4. Model giving the highest predicted CPI among all previous
        selected models is named as best max model.

        Args:
            column: CPI name.
        Returns:
             Validation metric and predictions of best max model.
        """
        models = []
        y = self.results[column][list(self.results[column].keys())[0]]["y"]
        for model_name in self.results[column].keys():
            if model_name != "best":
                error = calculate_metric(
                    self.results[column][model_name]["y"],
                    self.results[column][model_name]["y_predicted"],
                    self.evaluation_metric
                )
                models.append((model_name, error))
        models = sorted(models, key=lambda x: x[1], reverse=False)
        threshold = (1. + RANGE) * models[0][1]
        selected_models = [m[0] for m in models if m[1] <= threshold]
        max_model = []
        for ind in range(len(y)):
            predictions = [self.results[column][model_name]["y_predicted"][ind]
                           for model_name in self.results[column].keys()
                           if model_name in selected_models]
            max_model.append(max(predictions))
        error_max_model = calculate_metric(y,
                                           max_model,
                                           self.evaluation_metric)
        df = None
        for model_name in self.results[column].keys():
            if model_name in selected_models:
                if df is None:
                    df = pd.DataFrame(
                        self.results[column][model_name]["prediction"]
                    )
                else:
                    df = pd.concat(
                        [df, self.results[column][model_name]["prediction"]],
                        axis=1
                    )
        predict_max_model = df.max(axis=1).rename(column)
        return (error_max_model,
                predict_max_model)

    def evaluate_forecast(self) -> None:
        """ Create optimal models, evaluate, make prediction for all CPIs, and generate submissions.

        """
        prediction = {}
        mean_train_metric = {}
        mean_test_metric = {}
        mean_naive_metric = 0.
        # evaluate for each CPI
        for column in self.columns:
            x_columns = self.x_columns["X"] + self.x_columns[column]
            x = self.x_data.loc[:, x_columns]
            y = self.y_data.loc[:, column]
            evl_m = Evaluator(x,
                              y,
                              metric=self.evaluation_metric,
                              validation_steps=self.validation_steps,
                              forecast_steps=self.forecast_steps)
            naive_metric = evl_m.calculate_seasonal_naive_forecast_metric(12)
            mean_naive_metric += naive_metric
            evl_m.create_optimal_models(self.evaluation_models)
            for (model_name,
                 optimal_model,
                 test_score,
                 train_score,
                 predictions,
                 y_true) in evl_m.optimal_models:
                print(column,
                      model_name,
                      test_score,
                      train_score,
                      naive_metric
                      )
                mean_train_metric[model_name] = mean_train_metric.get(
                    model_name,
                    0.
                ) + train_score
                mean_test_metric[model_name] = mean_test_metric.get(
                    model_name,
                    0.
                ) + test_score
                # make prediction
                evl = Evaluator(x,
                                y,
                                self.evaluation_metric)
                y_predicted = evl.predict(optimal_model)
                y_extended = pd.concat([y, y_predicted])
                y_extended.rename(column, inplace=True)
                if model_name not in prediction.keys():
                    prediction[model_name] = pd.DataFrame(y_extended)
                    prediction[model_name].index.name = "date"
                else:
                    prediction[model_name] = pd.concat([prediction[model_name],
                                                        y_extended], axis=1)
                self.results.setdefault(column, {})
                self.results[column].setdefault(model_name, {})
                self.results[column][model_name] = {
                    "y_predicted": predictions,
                    "y": y_true,
                    "prediction": y_extended
                }
            (error_max_model,
             predict_max_model) = self.evaluate_best_models(column)
            if "best_max" not in prediction.keys():
                prediction["best_max"] = pd.DataFrame(predict_max_model)
                prediction["best_max"].index.name = "date"
            else:
                prediction["best_max"] = pd.concat([prediction["best_max"],
                                                    predict_max_model],
                                                   axis=1)
            print(column, "best_max", error_max_model, None, naive_metric)
            mean_train_metric["best_max"] = 0.
            mean_test_metric["best_max"] = mean_test_metric.get(
                "best_max",
                0.
            ) + error_max_model
        mean_naive_metric /= len(self.columns)
        self.evaluation_models.append("best")
        self.evaluation_models.append("best_max")
        for model_name in self.evaluation_models:
            mean_train_metric[model_name] /= len(self.columns)
            mean_test_metric[model_name] /= len(self.columns)
            print(model_name,
                  mean_test_metric[model_name],
                  mean_train_metric[model_name],
                  mean_naive_metric)
            self.y_data = prediction[model_name]
            self.__revert_y_data()
            if model_name.startswith("best"):
                self.generate_submissions(model_name)
