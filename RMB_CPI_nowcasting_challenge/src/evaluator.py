# -*- coding: utf-8 -*-
"""
The file contains python code to evaluate models.

Created on 26.09.2023

@author: ihar
"""

from typing import Any, Callable, Optional
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.linear_model import Lars, Lasso, Ridge, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from src.grid import grid
from src.sarimax import SARIMAX
from src.utils import calculate_metric
from settings import SEED, FEATURE_SELECTION


class Evaluator:
    def __init__(self,
                 x: pd.DataFrame,
                 y: pd.Series,
                 metric: str,
                 optimal_predictors: Optional[tuple[str]] = (),
                 validation_steps: Optional[int] = 0,
                 forecast_steps: Optional[int] = 1) -> None:
        """
        Args:
            x: X data.
            y: Y data.
            metric: Metric used for validation.
            optimal_predictors: Names of columns,
                                which can be used for selection some
                                optimal predictors from X data and use
                                this selected optimal X data for evaluation.
            validation_steps: Number of validation steps.
            forecast_steps:  Number of forecast steps.
        """
        self.optimal_models = []
        self.scaling = True
        self.validation_steps = validation_steps
        self.forecast_steps = forecast_steps
        self.metric = metric
        if self.validation_steps >= 2:
            self.tscv = TimeSeriesSplit(n_splits=self.validation_steps,
                                        test_size=self.forecast_steps)
        else:
            self.tscv = None
        # split data into train, test, and predict parts
        if self.validation_steps > 0:
            idx_train = x.index[:-self.validation_steps-self.forecast_steps]
            if self.forecast_steps > 0:
                idx_test = x.index[-self.validation_steps-self.forecast_steps:
                                   -self.forecast_steps]
            else:
                idx_test = x.index[
                           -self.validation_steps - self.forecast_steps:]
        else:
            if self.forecast_steps > 0:
                idx_train = x.index[:-self.forecast_steps]
            else:
                idx_train = x.index
            idx_test = x.index[:0]
        if self.forecast_steps > 0:
            if self.validation_steps > 0:
                idx_predict = idx_test.shift(
                    periods=self.forecast_steps
                )[-self.forecast_steps:]
            else:
                idx_predict = idx_train.shift(
                    periods=self.forecast_steps
                )[-self.forecast_steps:]
        else:
            idx_predict = x.index[:0]
        if not optimal_predictors:
            optimal_predictors = tuple(x.columns.to_list())
        self.x_train = x.loc[idx_train, optimal_predictors]
        self.x_test = x.loc[idx_test, optimal_predictors]
        self.x_predict = x.loc[idx_predict, optimal_predictors]
        self.y_train = y.loc[idx_train]
        self.y_test = y.loc[idx_test]
        self.__check_data()
        if self.scaling:
            if not self.x_train.empty:
                scaler = MinMaxScaler()
                scaler.fit(self.x_train)
                self.x_train[self.x_train.columns] = scaler.transform(
                    self.x_train
                )
                if not self.x_test.empty:
                    self.x_test[self.x_test.columns] = scaler.transform(
                        self.x_test
                    )
                if not self.x_predict.empty:
                    self.x_predict[self.x_predict.columns] = scaler.transform(
                        self.x_predict
                    )

    def __check_data(self):
        """Check integrity of x data.

        """
        assert not self.x_train.isnull().values.any(), \
            "Check for NaNs in X train"
        assert not self.x_test.isnull().values.any(), \
            "Check for NaNs in X test"
        assert not self.x_predict.isnull().values.any(), \
            "Check for NaNs in X predict"
        if not self.x_test.empty:
            assert max(self.x_train.index) < min(self.x_test.index), \
                "X test older than X train"
        if not self.x_predict.empty:
            if not self.x_test.empty:
                assert max(self.x_test.index) < min(self.x_predict.index), \
                    "X predict older than X test"
            else:
                assert max(self.x_train.index) < min(self.x_predict.index), \
                    "X predict older than X test"

    @staticmethod
    def metric_mapper(metric: str) -> tuple[str, Any]:
        """Map metric by its name.

        Args:
            metric: Metric name.

        Returns:
            Sklearn metric name and corresponding scorer.

        """
        mapping = {
            "mae": ("neg_mean_absolute_error", make_scorer(
                mean_absolute_error,
                greater_is_better=False)
                     ),
            "mape": ("neg_mean_absolute_percentage_error", make_scorer(
                mean_absolute_percentage_error,
                greater_is_better=False)
                     ),
            "mse": ("neg_mean_squared_error", make_scorer(
                mean_squared_error,
                greater_is_better=True,  # positive if True
                squared=True)
                     ),
            "rmse": ("neg_root_mean_squared_error", make_scorer(
                mean_squared_error,
                greater_is_better=True,  # positive if True
                squared=False)
                     ),
            "r2": ("r2", make_scorer(
                r2_score,
                greater_is_better=False)
                     )
        }
        return mapping[metric]

    def calculate_seasonal_naive_forecast_metric(
            self,
            seasonality: Optional[int] = 12) -> float:
        """Calculate seasonal naive forecast metric.

        Args:
            seasonality: Seasonality.

        Returns:
            Seasonal naive forecast metric.

        """
        seas_idx = self.y_test.index.shift(periods=-seasonality)
        y = pd.concat([self.y_train, self.y_test])
        return calculate_metric(self.y_test, y[seas_idx], self.metric)

    def create_default_model(self,
                             model_name: str,
                             **kwargs: Any) -> Optional[Any]:
        """Create default model by its name.

        Args:
            model_name: Model name.
            **kwargs:

        Returns:
            Model.

        """
        models = {
            "ab": AdaBoostRegressor(
                estimator=DecisionTreeRegressor(random_state=SEED),
                n_estimators=5,
                random_state=SEED,
                **kwargs
            ),
            "br": BaggingRegressor(
                estimator=DecisionTreeRegressor(random_state=SEED),
                n_estimators=200,
                random_state=SEED,
                **kwargs
            ),
            "cb": CatBoostRegressor(
                boosting_type="Plain",
                bootstrap_type="Bayesian",
                iterations=100,
                l2_leaf_reg=1,
                depth=2,
                leaf_estimation_iterations=1,
                has_time=True,
                random_state=SEED,
                logging_level="Silent",
                **kwargs
            ),
            "dt": DecisionTreeRegressor(
                **kwargs
            ),
            "knn": KNeighborsRegressor(
                **kwargs
            ),
            "lars": Lars(
                eps=1.e-5,
                random_state=SEED,
                **kwargs
            ),
            "lasso": Lasso(
                alpha=0.00005,
                max_iter=5000,
                **kwargs
            ),
            "lgbm": LGBMRegressor(
                learning_rate=0.05,
                extra_trees=True,
                random_state=SEED,
                **kwargs),
            "rf": RandomForestRegressor(
                n_estimators=70,
                random_state=SEED,
                **kwargs),
            "ridge": Ridge(
                alpha=0.1,
                **kwargs
            ),
            "sarimax": SARIMAX(),
            "sgd": SGDRegressor(
                penalty="elasticnet",
                loss="squared_error",
                learning_rate="constant",
                shuffle=False,
                random_state=SEED,
                **kwargs
            ),
            "svr": SVR(
                C=0.1,
                epsilon=0.001,
                kernel="sigmoid",
                **kwargs
            ),
            "xgb": XGBRegressor(
                    n_estimators=150,
                    learning_rate=0.06,
                    min_child_weight=10,
                    subsample=1,
                    random_state=SEED,
                    eval_metric=self.metric_mapper(self.metric)[1],
                    **kwargs
                )
        }
        try:
            model = models[model_name]
        except KeyError:
            model = None
        return model

    def create_optimal_model(self,
                             model_name: str,
                             n_jobs: Optional[int] = -1,
                             **kwargs: Any) -> tuple[Any, float]:
        """Create optimal model with grid search and calculate its train score.

        Args:
            model_name: Model name.
            n_jobs: Number of parallel jobs.
            **kwargs:

        Returns:
            Optimal model and its train score.

        """
        selector_name = FEATURE_SELECTION[model_name]
        model = self.create_default_model(model_name, **kwargs)
        if model_name == "sarimax":
            model.find_optimal_order(self.y_train)
            model.fit(self.x_train, self.y_train)
            best_estimator = model
            best_score = None
            print(model_name, model.order, model.seasonal_order, model.trend)
        else:
            if selector_name is not None:
                pipe = Pipeline(
                    [
                        ("selector", "passthrough"),
                        (model_name, model)
                    ]
                )
                estimator = pipe
                param_grid = grid[f"{model_name}_{selector_name}"].copy()
                n_features = self.x_train.shape[1]
                try:
                    param_grid["selector__k"] = [int(c * n_features)
                                                 for c in
                                                 param_grid["selector__k"]
                                                 if int(c * n_features) > 0]
                except KeyError:
                    pass
            else:
                estimator = model
                param_grid = grid[model_name].copy()
                if model_name == "lars":
                    n_features = self.x_train.shape[1]
                    param_grid["n_nonzero_coefs"] = [int(c * n_features)
                                                     for c in
                                                     param_grid[
                                                         "n_nonzero_coefs"]
                                                     if
                                                     int(c * n_features) > 0]
            grid_search = GridSearchCV(estimator=estimator,
                                       param_grid=param_grid,
                                       scoring=self.metric_mapper(
                                           self.metric
                                       )[0],
                                       n_jobs=n_jobs,
                                       cv=self.tscv,
                                       verbose=0)
            grid_search.fit(self.x_train, self.y_train)
            best_estimator = grid_search.best_estimator_
            best_score = -grid_search.best_score_
            print(model_name, grid_search.best_params_)
        return best_estimator, best_score

    def create_optimal_models(self, model_names: list[str]) -> None:
        """Create optimal models and evaluate it.

        Args:
            model_names: List of model names.

        """
        # create optimal models
        for model_name in model_names:
            optimal_model, _ = self.create_optimal_model(model_name)
            wfv = self.walk_forward_validation(optimal_model)
            self.optimal_models.append((model_name,
                                        optimal_model,
                                        wfv["test_score"],
                                        wfv["train_score"],
                                        wfv["predictions"],
                                        wfv["y"]))
        # sort optimal models and append best model
        self.optimal_models.sort(key=lambda x: x[2])
        best_model = self.optimal_models[0]
        self.optimal_models.insert(0, ("best",
                                       best_model[1],
                                       best_model[2],
                                       best_model[3],
                                       best_model[4],
                                       best_model[5]))

    def predict(self, model: Callable) -> pd.Series:
        """Make prediction.

        Args:
            model: Model is used for prediction.

        Returns:
            Prediction.

        """
        model.fit(self.x_train, self.y_train)
        y_predict = model.predict(self.x_predict)
        return pd.Series(data=y_predict,
                         index=self.x_predict.index,
                         dtype="float64")

    def walk_forward_validation(self,
                                model: Any,
                                x_t: Optional[pd.DataFrame] = None,
                                y_t: Optional[pd.Series] = None) -> dict:
        """Walk forward validation.

        Args:
            model: Model to test.
            x_t: x data for validation.
            y_t: y data for validation.

        Returns:
            Validation results.

        """
        if x_t is None:
            x_t = pd.concat([self.x_train, self.x_test])
        if y_t is None:
            y_t = pd.concat([self.y_train, self.y_test])
        ins_predictions = []
        oos_predictions = []
        for idx_train, idx_test in self.tscv.split(x_t):
            model.fit(x_t.iloc[idx_train, :], y_t.iloc[idx_train])
            ins_predictions.extend(model.predict(
                x_t.iloc[idx_train[-self.forecast_steps:], :])
            )
            oos_predictions.extend(model.predict(x_t.iloc[idx_test, :]))
        y_train = y_t.iloc[
                  -self.validation_steps-self.forecast_steps:
                  -self.forecast_steps
                  ].values
        y_test = y_t.iloc[-self.validation_steps:].values
        train_score = calculate_metric(y_train, ins_predictions, self.metric)
        test_score = calculate_metric(y_test, oos_predictions, self.metric)
        return {"train_score": train_score,
                "test_score": test_score,
                "predictions": oos_predictions,
                "y": y_test}
