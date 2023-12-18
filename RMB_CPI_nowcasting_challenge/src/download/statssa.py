# -*- coding: utf-8 -*-
"""
The file contains python code to load CPI data from Stats SA.

Created on 26.09.2023

@author: ihar
"""


from datetime import datetime
from typing import Optional
from io import BytesIO, TextIOWrapper
from zipfile import ZipFile
import requests
import pandas as pd
from dateutil.relativedelta import relativedelta
from src.download.web import WebLoader
from settings import MAPPING, URL_STATSSA_TS


class StatsSA(WebLoader):
    def __init__(self, end_month: datetime) -> None:
        """
        Args:
            end_month: Last month to load data.

        """
        super().__init__()
        self.end_month = end_month

    def __convert_to_df(self, cpi_list: list[dict]) -> pd.DataFrame:
        """Select required CPI components and convert it to DataFrame.
        Required columns are defined as keys in `MAPPING`.

        Args:
            cpi_list: List with all CPI components.

        Returns:
            DataFrame with required CPI components.

        """
        df = None
        columns = MAPPING.keys()
        columns = [col for col in columns if col[-3:] == "000"]
        for cpi in cpi_list:
            if cpi["H03"] in columns:
                ser = pd.Series(
                    data=cpi["data"],
                    index=self.__create_index(cpi["H24"], len(cpi["data"])),
                    name=cpi["H03"])
                if df is None:
                    df = ser
                else:
                    df = pd.concat([df, ser], axis=1)
        return df

    @staticmethod
    def __create_index(start: str, n_elements: int) -> pd.Index:
        """Create index of CPI component.

        Args:
            start: First (oldest) date of CPI component.
            n_elements: Length of CPI component.

        Returns:
            Index.

        """
        start = start.split(":")[1].strip()
        start = datetime.strptime(start, "%Y %m") + relativedelta(day=31)
        index = pd.date_range(start,
                              periods=n_elements,
                              freq="M")
        index.name = "date"
        return index

    def __load_cpi_file(self) -> Optional[BytesIO]:
        """Load zipped file with all CPI components from Stats SA.

        Returns:
            Zipped file with all CPI components.

        """
        cpi_file = None
        end_month = self.end_month.strftime("%Y%m")
        response = self.session.get(URL_STATSSA_TS.format(end_month))
        if response.status_code == requests.codes["ok"]:
            cpi_file = BytesIO(response.content)
        return cpi_file

    @staticmethod
    def __parse_file(file: BytesIO) -> list[dict]:
        """Unzip file and parse it.

        Args:
            file: Zipped file with all CPI components.

        Returns:
            List with all CPI components.

        """
        with ZipFile(file) as zip_file:
            with TextIOWrapper(zip_file.open(zip_file.infolist()[0].filename),
                               encoding="utf-8") as f:
                parsed_list = []
                parsed_dict = {}
                data = []
                for line in f:
                    line = line.rstrip("\n")
                    line = line.split(":", 1)
                    if len(line) > 1:
                        if line[0].strip() == "H01":
                            if parsed_dict:
                                parsed_dict["data"] = data
                                parsed_list.append(parsed_dict)
                            parsed_dict = {}
                            data = []
                        key = line[0].strip()
                        value = line[1].strip()
                        parsed_dict[key] = value
                    else:
                        data.append(float(line[0].strip()))
                if parsed_dict:
                    parsed_dict["data"] = data
                    parsed_list.append(parsed_dict)
                return parsed_list

    def load_cpi_data(self) -> pd.DataFrame:
        """Load required CPI components from Stats SA.

        Returns:
            DataFrame with required CPI components.

        """
        cpi_file = self.__load_cpi_file()
        parsed_file = self.__parse_file(cpi_file)
        df = self.__convert_to_df(parsed_file)
        return df
