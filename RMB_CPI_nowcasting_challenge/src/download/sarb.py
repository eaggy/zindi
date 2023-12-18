# -*- coding: utf-8 -*-

"""
The file contains python code to download different data from SARB.

Created on 26.09.2023

@author: ihar
"""


import json
from typing import Optional
from datetime import datetime
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from src.download.web import WebLoader
from settings import URL_API, URL_DOWNLOAD_FACILITY


class SARB(WebLoader):
    def __init__(self,
                 start_month: datetime,
                 end_month: datetime,
                 data_codes: list[str],
                 api_mapping: dict[str:str]) -> None:
        """
        Args:
            start_month: First month to load data.
            end_month: Last month to load data.
            data_codes: List of SARB data codes.
            api_mapping: Mapping between SARB data codes and corresponding SARB time series codes.

        """
        super().__init__()
        self.start_month = start_month
        self.end_month = end_month
        self.data_codes = data_codes
        self.api_mapping = api_mapping
        self.county_code = ""
        self.data_dimensions = []
        self.data_flow = []
        self.db_id = ""
        self.codes = {}

    def __load_data_api(self, ts_code: str) -> Optional[pd.DataFrame]:
        """Load single time series with SARB Web API for a particular time series code.

        Args:
            ts_code: SARB time series code.

        Returns:
            DataFrame with SARB time series.

        """
        df = None
        data = []
        start_date = self.start_month + relativedelta(day=1)
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = self.end_month + relativedelta(months=1) + relativedelta(day=31)
        end_date = end_date.strftime("%Y-%m-%d")
        response = self.session.get(f"{URL_API}{ts_code}/{start_date}/{end_date}/",)
        if response.status_code == requests.codes["ok"]:
            js = json.loads(response.text)
            try:
                data = [{"date": t["Period"],
                         ts_code: float(t["Value"])} for t in js]
            except KeyError:
                print(f"API: fail to load {ts_code}")
        if data:
            df = pd.DataFrame(data)
            df = self.__set_date_index(df, "%Y-%m-%dT%H:%M:%S")
            df = df.astype("float64")
            df = df.groupby(pd.Grouper(freq="M")).mean()
        return df

    def __load_data_query(self, data_code: str) -> Optional[pd.DataFrame]:
        """Load single time series with SARB online query facility for a particular SARB data code.

        Args:
            data_code: SARB data code in format KBP*****.

        Returns:
            DataFrame with SARB time series.

        """
        df = None
        data = []
        if data_code.endswith("K") or data_code.endswith("L"):
            frequency = "Quarterly"
            quarter = str((int(self.start_month.strftime("%m")) - 1) // 3 + 1)
            year = self.start_month.strftime("%Y")
            start_date = f"{year}/{quarter}"
            quarter = str((int(self.end_month.strftime("%m")) - 1) // 3 + 1)
            year = self.end_month.strftime("%Y")
            end_date = f"{year}/{quarter}"
        elif data_code.endswith("W"):
            frequency = "Weekly (5 days)"
            week = str((int(self.start_month.strftime("%d")) - 1) // 7 + 1)
            year = self.start_month.strftime("%Y")
            month = self.start_month.strftime("%m")
            start_date = f"{year}/{month}/{week}"
            week = str((int(self.end_month.strftime("%d")) - 1) // 7 + 1)
            year = self.end_month.strftime("%Y")
            month = self.end_month.strftime("%m")
            end_date = f"{year}/{month}/{week}"
        else:
            frequency = "Monthly"
            start_date = self.start_month.strftime("%Y/%m")
            end_date = self.end_month.strftime("%Y/%m")
        payload = {"onlineDownload": "sSRSData",
                   "sSRSDataTsCodes": data_code,
                   "sSRSDataFrequencyDescription": frequency,
                   "sSRSDataStartDate": start_date,
                   "sSRSDataEndDate": end_date}
        response = self.session.get(URL_DOWNLOAD_FACILITY, params=payload)
        if response.status_code == requests.codes["ok"]:
            js = json.loads(response.text)
            try:
                table = js["xs:ssrsDataResult"]["diffgr:diffgram"]["TsObservations"]["Table"]
                data = [{"date": str(t["Period"])[:-2],
                         data_code: float(t["Value"])} for t in table]
            except KeyError:
                print(f"Query: fail to load {data_code}")
        if data:
            df = pd.DataFrame(data)
            if frequency == "Quarterly":
                df.loc[:, "date"] = df.loc[:, "date"].apply(
                    lambda s: f"{s[:4]}-Q{int(s[-2:]):}"
                )
                df = self.__set_date_index(df, "%Y%q")
                df = df.resample("M").bfill()
                shifted_index = df.index.shift(periods=2)
                df = df.set_axis(shifted_index)
                index = df.index.union(
                    pd.date_range(start=df.index[0] - 2 * df.index.freq,
                                  periods=2,
                                  freq=df.index.freq,
                                  name="date")
                )
                first_value = df.loc[:, data_code].iloc[0]
                values = 2*[first_value] + list(df.loc[:, data_code])
                df = pd.DataFrame(data=values,
                                  index=index,
                                  columns=[data_code])
            elif frequency == "Weekly (5 days)":
                df = self.__set_date_index(df, "%Y%m")
                df = df.groupby(pd.Grouper(freq="M")).mean()
            else:
                df = self.__set_date_index(df, "%Y%m")
            df = df.astype("float64")
        return df

    @staticmethod
    def __set_date_index(df: pd.DataFrame,
                         date_format: Optional[str] = "") -> pd.DataFrame:
        """Set date index in DataFrame.

        Args:
            df: DataFrame to set date index.
            date_format: Format of date.

        Returns:
            DataFrame with set date index.

        """
        if "%q" in date_format:
            df["date"] = pd.PeriodIndex(df["date"], freq="Q").to_timestamp()
            df["date"] = df["date"].apply(lambda d: d + relativedelta(day=31))
        elif "%d" in date_format:
            df["date"] = pd.to_datetime(df["date"], format=date_format)
        else:
            df["date"] = pd.to_datetime(df["date"], format=date_format)
            df["date"] = df["date"].apply(lambda d: d + relativedelta(day=31))
        df.set_index(["date"], inplace=True)
        df.sort_index(inplace=True)
        return df

    def load_sarb_data(self) -> Optional[pd.DataFrame]:
        """Load data using SARB online query facility and/or SARB Web API.

        Returns:
            DataFrame with SARB data.

        """
        all_data = None
        for data_code in self.data_codes:
            if data_code in self.api_mapping.keys():
                data = self.__load_data_api(self.api_mapping[data_code])
                if data is not None:
                    data.rename(
                        columns={self.api_mapping[data_code]: data_code},
                        inplace=True
                    )
            else:
                data = self.__load_data_query(data_code)
                if data is None:
                    # try to load using API
                    ts_code = data_code.replace("KBP", "BOP")
                    data = self.__load_data_api(ts_code)
                    if data is not None:
                        data.rename(
                            columns={ts_code: data_code},
                            inplace=True
                        )
            if data is not None:
                if all_data is None:
                    all_data = data
                else:
                    all_data = pd.concat([all_data, data], axis=1)
        return all_data
