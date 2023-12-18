# -*- coding: utf-8 -*-
"""
The file contains python code to transform data.

Created on 26.09.2023

@author: ihar
"""

from datetime import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
from src.utils import get_d_sd, extend_index


class Transformer:
    def __init__(self,
                 start_month: str,
                 end_month: str,
                 forecast_steps: int) -> None:
        """
        Args:
            start_month: First month to load data in format "%Y-%m".
            end_month: Last month to load data in format "%Y-%m".
            forecast_steps: Number of forecast steps.

        """
        self.transformations = ["log", "pct"]
        self.start_month = datetime.strptime(
            start_month, "%Y-%m") + relativedelta(day=31)
        self.end_month = datetime.strptime(
            end_month, "%Y-%m") + relativedelta(day=31)
        self.forecast_steps = forecast_steps

    def __shift(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shift each column of time sorted Dataframe if this column
           has no values for the latest time points (NaNs rows at the end of column)
           till these NaNs rows will be gone. If shift is applied to the column,
           the suffix "_lag{n}" is added to the column name, where {n} is number
           of the applied shift steps.

        Args:
            df: DataFrame to shift.

        Returns:
            Shifted DataFrame.

        """
        # add new rows for forecast
        max_forecast_date = self.end_month + relativedelta(
            months=self.forecast_steps
        )
        extra_periods = relativedelta(max_forecast_date, max(df.index)).months
        if extra_periods > 0:
            df = pd.DataFrame(
                index=extend_index(df, extra_periods)
            ).join(df)
            df.sort_index(inplace=True)
        # shift each column till NaNs at the end of the column will be gone
        for column in df.columns.to_list():
            lag = df.loc[:, column].isnull().astype(int).groupby(
                df.loc[:, column].notnull().astype(int).cumsum()
            ).sum().iloc[-1]
            df.loc[:, column] = df.loc[:, column].shift(periods=lag)
            df.rename(columns={column: f"{column}_lag{lag}"}, inplace=True)
        return df

    def transform_sarb(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform SARB data.
           The following transformations are applied to time series in each column:
           1. If time series is positive, the difference to the previous month is calculated.
           2. Tests for seasonality and stationarity are conducted to find most
              probable values of "D" and "d".
           3. If "D" and/or "d" > 0, the seasonal differencing of order "D"
              and/or differencing of order "d" are applied to the time series.

        Args:
            df: DataFrame to apply transformation.

        Returns:
            Transformed DataFrame.

        """
        df.sort_index(inplace=True)
        periods = relativedelta(self.end_month, max(df.index)).months
        if periods > 0:
            df = pd.DataFrame(
                index=extend_index(df, periods)
            ).join(df)
        for column in df.columns.to_list():
            ser = df.loc[:, column]
            ser.dropna(inplace=True)
            if min(ser) > 0.:
                ser = ser.diff(periods=1)
                ser.dropna(inplace=True)
            d, sd = get_d_sd(ser)
            while d or sd:
                if sd:
                    ser = ser.diff(periods=12)
                    ser.dropna(inplace=True)
                    d, sd = get_d_sd(ser)
                if d:
                    ser = ser.diff(periods=1)
                    ser.dropna(inplace=True)
                    d, sd = get_d_sd(ser)
            df.loc[:, column] = ser
        df = self.__shift(df)
        return df
