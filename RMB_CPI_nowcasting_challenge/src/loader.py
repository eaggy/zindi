# -*- coding: utf-8 -*-
"""
The file contains python code for loading and preparation of X and Y data.

Created on 26.09.2023

@author: ihar
"""

from typing import Union
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from src.transformer import Transformer
from src.utils import generate_file_path
from settings import STATSSA_FILE_LOCATION, SARB_FILE_LOCATION


class Loader(Transformer):
    """Load data from different CSV-files and transform it."""

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
        super().__init__(start_month, end_month, forecast_steps)
        self.start_month = datetime.strptime(
            start_month, "%Y-%m") + relativedelta(day=31)
        self.end_month = datetime.strptime(
            end_month, "%Y-%m") + relativedelta(day=31)

    def __add_cpi_data(self, df: Union[pd.DataFrame, None]) -> pd.DataFrame:
        """Add CPI data from csv-file to DataFrame.

        Args:
            df: DataFrame to add CPI data.

        Returns:
             DataFrame with added CPI data.

        """
        file_path = generate_file_path(STATSSA_FILE_LOCATION,
                                       self.end_month.strftime("%Y%m"))
        cpi_data = pd.read_csv(file_path,
                               index_col="date",
                               date_parser=lambda s: datetime.strptime(
                                   s,
                                   "%Y-%m-%d"
                               ),
                               encoding="utf-8")
        cpi_data.index.freq = "M"
        cpi_data = cpi_data.astype("float64")
        if df is None:
            cpi_data.index.name = "date"
            df = cpi_data.loc[:, :]
        else:
            df = pd.concat([df, cpi_data], axis=1)
        return df

    def __add_sarb_data(self, df: Union[pd.DataFrame, None]) -> pd.DataFrame:
        """Add SARB data from csv-file to DataFrame.

         Args:
            df: DataFrame to add SARB data.

        Returns:
            DataFrame with added SARB data.

        """
        file_path = generate_file_path(SARB_FILE_LOCATION,
                                       self.end_month.strftime("%Y%m"))
        sarb_data = pd.read_csv(file_path,
                                index_col="date",
                                date_parser=lambda s: datetime.strptime(
                                    s,
                                    "%Y-%m-%d"
                                ),
                                encoding="utf-8")
        sarb_data.index.freq = "M"
        sarb_data = sarb_data.astype("float64")
        sarb_data = self.transform_sarb(sarb_data)
        if df is None:
            sarb_data.index.name = "date"
            df = sarb_data.loc[:, :]
        else:
            df = pd.concat([df, sarb_data], axis=1)
        return df

    def load_all_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load all (SARB and CPI) data from csv-files.

        Returns:
            DataFrame with all data.

        """
        all_data = None
        # load data from SARB
        all_data = self.__add_sarb_data(all_data)
        # load cpi data
        # IMPORTANT: this must be the last operation
        all_data = self.__add_cpi_data(all_data)
        # separate x (SARB data) and y(cpi data) columns
        columns = list(all_data.columns)
        x_columns = [col for col in columns if not col.startswith("CPS")]
        y_columns = [col for col in columns if col.startswith("CPS")]
        # select time range
        x_data = pd.DataFrame()
        end_month = self.end_month + relativedelta(months=self.forecast_steps,
                                                   day=31)

        try:
            x_data = all_data.loc[
                (all_data.index.get_level_values("date") >= self.start_month) &
                (all_data.index.get_level_values("date") <= end_month),
                x_columns]
        except KeyError:
            pass
        y_data = all_data.loc[
            (all_data.index.get_level_values("date") <= self.end_month),
            y_columns]
        assert not y_data.empty, "y_data is empty"
        assert not y_data.isnull().values.any(), "Check for NaNs in y_data"
        return x_data, y_data
