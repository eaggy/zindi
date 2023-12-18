# -*- coding: utf-8 -*-
"""
The file contains python code of SARIMAX model.

Created on 26.09.2023

@author: ihar
"""

import itertools
from typing import Any, Optional
import multiprocessing as mp
from joblib import Parallel, delayed
import pandas as pd
from pmdarima import ARIMA
from src.grid import grid
from src.utils import get_d_sd
from settings import IC


class SARIMAX(ARIMA):
    def __init__(self,
                 order: Optional[tuple[int, int, int]] = (0, 0, 0),
                 seasonal_order: Optional[tuple[int, int, int, int]] = (0, 0, 0, 0),
                 information_criterion: Optional[str] = "hqic",
                 filter_x: Optional[bool] = True,
                 trend: Optional[str] = "n",
                 method: Optional[str] = "powell",
                 maxiter: Optional[int] = 5000,
                 suppress_warnings: Optional[bool] = True,
                 enforce_stationarity: Optional[bool] = False,
                 enforce_invertibility: Optional[bool] = False
                 ) -> None:
        """
        Args:
            order: The (p,d,q) order of the model.
            seasonal_order: The (P,D,Q,s) order of the seasonal component of the model.
            information_criterion: Information criterion used for model optimization.
            filter_x: Filter X data if True.
            trend: Parameter controlling the deterministic trend polynomial.
                   Can be specified as a string where "n" indicates no trend
                   "c" indicates a constant, "t" indicates a linear trend
                   with time, and "ct" is both.
            method: The method determines which solver from scipy.optimize is
                    used, and it can be chosen from among the following strings:
                        "newton" for Newton-Raphson
                        "nm" for Nelder-Mead
                        "bfgs" for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
                        "lbfgs" for limited-memory BFGS with optional box constraints
                        "powell" for modified Powell’s method
                        "cg" for conjugate gradient
                        "ncg" for Newton-conjugate gradient
                        "basinhopping" for global basin-hopping solver.
            maxiter: Max number of iterations  to perform.
            suppress_warnings: Suppress warnings if True.
            enforce_stationarity: Enforce_stationarity if True.
            enforce_invertibility: Enforce invertibility if True.

        """
        super().__init__(order=order,
                         seasonal_order=seasonal_order,
                         trend=trend,
                         start_params=None,
                         method=method,
                         maxiter=maxiter,
                         suppress_warnings=suppress_warnings,
                         out_of_sample_size=0,
                         scoring="mse",
                         scoring_args=None,
                         with_intercept=True,
                         enforce_stationarity=enforce_stationarity,
                         enforce_invertibility=enforce_invertibility)
        self.information_criterion = information_criterion
        self.filter_x = filter_x
        self._method = method
        self._maxiter = maxiter

    @staticmethod
    def __filter_x(x: Any) -> Any:
        """Fiter X data by selecting only COVID and dummy columns.

        Args:
            x: DataFrame with X data.

        Returns:
             Filtered X data.

        """
        if isinstance(x, pd.DataFrame):
            columns = x.columns.to_list()
            columns = [c for c in columns if not (
                    c.startswith("KBP") or
                    c.startswith("lag_")
            )]
            x = x.loc[:, columns]
        return x

    @staticmethod
    def __generate_sarima_params(y: pd.Series, m: Optional[int] = 12) -> list[
        tuple[
            tuple[int, int, int],
            tuple[int, int, int, int],
            str
        ]
    ]:
        """Generate of list with all possible combinations SARIMA parameters.
        SARIMA parameters "p", "q", "P", "Q", and "trend" are defined by grid.
        Parameters "d" and "D" are defined by statistical tests.

        Args:
            y: Y time series.
            m: Periodicity.

        Returns:
            List with all possible combinations SARIMA parameters.

        """
        sarima_grid = grid["sarimax"]
        sarima_params = list(itertools.product(
            sarima_grid["p"],
            sarima_grid["q"],
            sarima_grid["P"],
            sarima_grid["Q"],
            sarima_grid["trend"]
        ))
        d, sd = get_d_sd(y)
        sarima_params = [((c[0], d, c[1]), (c[2], sd, c[3], m), c[4])
                         for c in sarima_params]
        return sarima_params

    @staticmethod
    def __optimization_worker(y: pd.Series,
                              order: tuple[int, int, int],
                              seasonal_order: tuple[int, int, int, int],
                              trend: str,
                              method: str,
                              maxiter: int) -> tuple[float, float, float]:
        """Fit SARIMA model for given `order`, `seasonal_order`, and `trend`
           and calculate its information criteria.

        Args:
            y: Y data. The time series to which to fit the SARIMA estimator.
            order: The (p,d,q) order of the model.
            seasonal_order: The (P,D,Q,s) order of the seasonal component of the model.
            trend: Parameter controlling the deterministic trend polynomial.
                   Can be specified as a string where "n" indicates no trend
                   "c" indicates a constant, "t" indicates a linear trend
                   with time, and "ct" is both.
            method: The method determines which solver from scipy.optimize is
                    used, and it can be chosen from among the following strings:
                        "newton" for Newton-Raphson
                        "nm" for Nelder-Mead
                        "bfgs" for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
                        "lbfgs" for limited-memory BFGS with optional box constraints
                        "powell" for modified Powell’s method
                        "cg" for conjugate gradient
                        "ncg" for Newton-conjugate gradient
                        "basinhopping" for global basin-hopping solver.
            maxiter: Max number of iterations  to perform.

        Returns:
            "aic", "bic", and  "hqic" information criteria for the fitted SARIMA model.

        """
        model = ARIMA(
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            method=method,
            maxiter=maxiter,
            suppress_warnings=False,
            with_intercept=True,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(y)
        return model_fit.aic(), model_fit.bic(), model_fit.hqic()

    def find_optimal_order(self, y: pd.Series, n_jobs: int = -1):
        """Find optimal SARIMA parameters using grid search for
           the given information criterion `IC`. These parameters must minimize
           the value of information criterion.

        Args:
            y: Y data. The time series to which to fit the SARIMA estimator.
            n_jobs: Number of models to fit in parallel.

        """
        ic_name_mapping = {"aic": 0,
                           "bic": 1,
                           "hqic": 2}
        search_results = []
        sarima_params = self.__generate_sarima_params(y)
        if n_jobs == 1:
            for params in sarima_params:
                ic = self.__optimization_worker(y,
                                                order=params[0],
                                                seasonal_order=params[1],
                                                trend=params[2],
                                                method=self._method,
                                                maxiter=self._maxiter)
                search_results.append((ic, params))
        else:
            n_cpu = mp.cpu_count()
            n_jobs = min(n_jobs, n_cpu)
            if n_jobs < 1:
                n_jobs = n_cpu
            results = Parallel(n_jobs=n_jobs)(delayed(
                self.__optimization_worker
            )(
                y,
                params[0],
                params[1],
                params[2],
                self._method,
                self._maxiter
            ) for params in sarima_params)
            search_results = list(zip(results, sarima_params))
        best_search_result = min(search_results,
                                 key=lambda x: x[0][ic_name_mapping[IC]])
        optimal_parameters = best_search_result[1]
        self.order = optimal_parameters[0]
        self.seasonal_order = optimal_parameters[1]
        self.trend = optimal_parameters[2]

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        """Fit model.

        Args:
            x: X data.
            y: Y data.

        """
        if self.filter_x:
            x = self.__filter_x(x)
        super().fit(y, X=x)

    def predict(self, x: pd.DataFrame) -> Any:
        """Make prediction.

        Args:
            x: X data for prediction.

        Returns:
             Predicted Y data.

        """
        idx_list = []
        for idx in x.index.values:
            try:
                idx_list.append(self.endog_index_.get_loc(idx))
            except KeyError:
                pass
        if self.filter_x:
            x = self.__filter_x(x)
        if idx_list:
            prediction = super().predict_in_sample(start=min(idx_list),
                                                   end=max(idx_list),
                                                   X=x)
        else:
            prediction = super().predict(n_periods=len(x.index), X=x)
        return prediction
