# -*- coding: utf-8 -*-
"""
The file contains python code to download X (SARB) and Y (CPI) data and save as csv.

Created on 26.09.2023

@author: ihar
"""

import os.path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from src.download.sarb import SARB
from src.download.statssa import StatsSA
from src.utils import generate_file_path
from settings import START_MONTH, END_MONTH
from sarb import API_MAPPING, data_codes
from settings import SARB_FILE_LOCATION, STATSSA_FILE_LOCATION

if __name__ == "__main__":
    start_month = datetime.strptime(
        START_MONTH, "%Y-%m"
    ) + relativedelta(day=31)
    end_month = datetime.strptime(
        END_MONTH, "%Y-%m"
    ) + relativedelta(day=31)

    # download SARB data
    sarb_file_path = generate_file_path(
        SARB_FILE_LOCATION,
        end_month.strftime("%Y%m"))
    if not os.path.exists(sarb_file_path):
        sarb = SARB(start_month, end_month, data_codes, API_MAPPING)
        try:
            sarb_data = sarb.load_sarb_data()
            sarb_data.to_csv(sarb_file_path, encoding="utf-8")
        except KeyError:
            print("Fail to download data from SARB.")

    # download CPI data
    cpi_file_path = generate_file_path(
        STATSSA_FILE_LOCATION,
        end_month.strftime("%Y%m"))
    if not os.path.exists(cpi_file_path):
        statssa = StatsSA(end_month=end_month)
        try:
            cpi_data = statssa.load_cpi_data()
            cpi_data.to_csv(cpi_file_path, encoding="utf-8")
        except AttributeError:
            print("Fail to download data from StatsSA.")
