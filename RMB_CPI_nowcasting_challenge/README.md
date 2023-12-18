## Getting Started
* Unzip project.
* Go to project folder:

    `cd RMB_CPI_nowcasting_challenge`

* Create a new conda virtual environment:

    `conda env create -f environment.yml`

* Activate it:

    `conda activate cpi`

* Set the proper value of `END_MONTH` in the file `settings.py`.
To predict CPI for September 2023 this value must be "2023-08" (one month before).
To predict October 2023 - "2023-09", November 2023 - "2023-10", and so on.

* Run script:

    `python forecaster.py`

    to make predictions and generate submission CSV-files in the folder `RMB_CPI_nowcasting_challenge/data/output/`.
The script generates four files:
1. `submission_best_{month}.csv`, where {month} is the forecasting month (September, October, or November). The monthly forecast of the best model (see below) for all 13 CPI components for this month.
2. `submission_best_max_{month}.csv`, where {month} is the forecasting month (September, October, or November). The monthly forecast of the best max model (see below) for all 13 CPI components for this month.
3. `submission_best_3m.csv` - the forecast of the best model (see below) for all 39 (3x13) CPI components for the months September, October, and November
This file is built using the monthly forecast files of the best model for
September, October, and November:`submission_best_September.csv`,`submission_best_October.csv`,`submission_best_November.csv`.
If monthly forecast files do not yet exist, zero values will be used to build this file. Therefore, it is important to keep
all already generated monthly forecast files in this folder to ensure,
that this file will be properly populated with CPI values from the monthly forecast files.
4. `submission_best_max_3m.csv` - the forecast of the best max model (see below) for all 39 (3x13) CPI components for the months September, October, and November
This file is built using the monthly forecast files of the best max model for
September, October, and November:`submission_best_max_September.csv`,`submission_best_max_October.csv`,`submission_best_max_November.csv`.
If monthly forecast files do not yet exist, zero values will be used to build this file. Therefore, it is important to keep
all already generated monthly forecast files in this folder to ensure,
that this file will be properly populated with CPI values from the monthly forecast files.

The files `submission_best_3m.csv` and `submission_best_max_3m.csv` were submitted to Zindi.

The running time of this script on HP ZBook 15 G3 (Intel(R) Core(TM) i7-6820HQ CPU @ 2.70GHz) is **70 minutes**.

This script uses data (see below) downloaded from Statistics South Africa StatsSA (https://www.statssa.gov.za/)
and South African Reserve Bank SARB (https://www.resbank.co.za/). The data are saved in the folder `RMB_CPI_nowcasting_challenge/data/` as
CSV-files `statssa_2023{mm}.csv` and `sarb_2023{mm}.csv` respectively, where {mm} is 08 to forecast CPI for September,
09 to forecast CPI for October, and 10 to forecast CPI for November. The newest versions of these files can be downloaded by running the script:

`python downloader.py`

This script does not overwrite CSV-files automatically.
The existing CSV-files `statssa_2023{mm}.csv` and `sarb_2023{mm}.csv` must be deleted manually before run the script.

**IMPORTANT! Do not run this script during the validation of the submitted solution.
Because with the newest downloaded data slightly different CPI values could be predicted.
Instead, use the provided files, which were downloaded on the submission day.**


## Project Structure
The project is organized as follows:
````
RMB_CPI_nowcasting_challenge
├── data
│   ├── output
│   │   ├── submission_best_3m.csv
│   │   ├── submission_best_max_3m.csv
│   │   ├── submission_best_max_November.csv
│   │   ├── submission_best_max_October.csv
│   │   ├── submission_best_max_September.csv
│   │   ├── submission_best_November.csv
│   │   ├── submission_best_October.csv
│   │   └── submission_best_September.csv
│   ├── sarb_202308.csv
│   ├── sarb_202309.csv
│   ├── sarb_202310.csv
│   ├── statssa_202308.csv
│   ├── statssa_202309.csv
│   └── statssa_202310.csv
├── src
│   ├── download
│   │   ├── sarb.py
│   │   ├── statssa.py
│   │   └── web.py
│   ├── evaluator.py
│   ├── grid.py
│   ├── loader.py
│   ├── model.py
│   ├── sarimax.py
│   ├── transformer.py
│   └── utils.py
├── downloader.py
├── environment.yml
├── forecast.py
├── README.md
├── sarb.py
└── settings.py
````
The folder `RMB_CPI_nowcasting_challenge/data/`:

`sarb_202308.csv` - file with data downloaded from SARB and containing various time series (see below) used as predictors (X data) to make CPI prediction for September 

`sarb_202309.csv` - file with data downloaded from SARB and containing various time series (see below) used as predictors (X data) to make CPI prediction for October 

`sarb_202310.csv` - file with data downloaded from SARB and containing various time series (see below) used as predictors (X data) to make CPI prediction for November 

`statssa_202308.csv` - file with data downloaded from StatsSA and containing historical data of all 13 CPI components (Y data) to make CPI prediction for September

`statssa_202309.csv` - file with data downloaded from StatsSA and containing historical data of all 13 CPI components (Y data) to make CPI prediction for October

`statssa_202310.csv` - file with data downloaded from StatsSA and containing historical data of all 13 CPI components (Y data) to make CPI prediction for November

**IMPORTANT! Do not overwrite these files. Because with the newest downloaded data slightly different CPI values could be predicted.**

The folder `RMB_CPI_nowcasting_challenge/data/output/`:

`submission_best_3m.csv` - CPI prediction for September, October, and November from the best model (see below). **This file was submitted to Zindi.**

`submission_best_max_3m.csv` - CPI prediction for September, October, and November from the best max model (see below). **This file was submitted to Zindi.**

`submission_best_max_November.csv` - monthly CPI prediction for November from the best max model (see below)

`submission_best_max_October.csv` - monthly CPI prediction for October from the best max model (see below)

`submission_best_max_September.csv` - monthly CPI prediction for September from the best max model (see below)

`submission_best_November.csv` - monthly CPI prediction for November from the best model (see below)

`submission_best_October.csv` - monthly CPI prediction for October from the best model (see below)

`submission_best_September.csv` - monthly CPI prediction for September from the best model (see below)

The folder `RMB_CPI_nowcasting_challenge/src/download/`:

`sarb.py` - python class to download various time series (see below) from SARB and save in CSV-file

`statssa.py` - python class to download historical data of all 13 CPI components from StatsSA and save in CSV-file

`web.py` - python class of basic web downloader

The folder `RMB_CPI_nowcasting_challenge/src/`:

`evaluator.py` - python class for models evaluation and forecasting

`grid.py` - grid definitions for hyperparameters tuning

`loader.py` - python class for loading from CSV-files and preparation of X and Y data

`model.py` - main python class to create models and make predictions of CPI

`sarimax.py` - python class for SARIMAX model

`transformer.py` - python class for data transformation

`utils.py` - some useful functions

The folder `RMB_CPI_nowcasting_challenge/`:

`downloader.py` - main script to download the newest SARB and StatsSA data from the internet and save it as CSV-files. **IMPORTANT! Do not run this script during the verification of the submitted solution. Instead, use the provided files, which were downloaded on the submission day.**

`environment.yml` - conda environment

`forecast.py` - main script to predict CPI and generate submission files. This script must be executed.

`README.md` - this file

`sarb.py` - file contains codes and a description of SARB data

`settings.py` - important project settings. It is only necessary to change `END_MONTH` to predict CPI for the new month (see above).

## Data Sources
The following external data sources are used:

* StatsSA data - historical CPI data from this page
https://www.statssa.gov.za/?page_id=1847 Timeseries File P0141 - CPI(COICOP) from Jan 2008 (202310).zip in ASCII format.
This file can be downloaded with `RMB_CPI_nowcasting_challenge/downloader.py` and saved as CSV-file in `RMB_CPI_nowcasting_challenge/data/`.

* SARB data - various time series from SARB online statistical query: https://www.resbank.co.za/en/home/what-we-do/statistics/releases/online-statistical-query.
The SARB codes of the time series to download are defined in `RMB_CPI_nowcasting_challenge/sarb.py` by `data_codes`.
Descriptions of time series are provided as comments in the same file.
Some time series are delivered with quarterly frequency. These series are downsampled to a monthly frequency.
It is possible to download some time series with daily frequency using SARB Web API: https://custom.resbank.co.za/SarbWebApi/
The API codes of such series are defined as values of dictionary `API_MAPPING` in `RMB_CPI_nowcasting_challenge/sarb.py`.
This dictionary also maps the SARB codes from online statistical query with the API codes.
The daily time series defined in `API_MAPPING` are downloaded using the SARB Web API and
then upsampled to monthly frequency using the averaging over the entire month.
The SARB data can be downloaded from the SARB statistical query and SARB Web API
with `RMB_CPI_nowcasting_challenge/downloader.py` and saved as CSV-file in `RMB_CPI_nowcasting_challenge/data/`.

## How It Works
* Historical CPI data (StatsSA data) are used as Y data. All models created here use only
one CPI component as a dependent variable of the model. Therefore, there is a loop over all 13 CPI components.
   * The following two transformations are applied to the Y data:
      1. Logarithm-transformation,
      2. Difference to the previous month.
    
  At the end, the inverse transformations are applied to predicted data to get normal predicted CPI values.

* SARB time series are used as X data.
   * The following two transformations are applied to these data:
      1. Difference to the previous month, if all values in the time series are positive.
      2. Tests for seasonality (CH test and OCSB test) and stationarity (ADF test, KPSS test, and PP test) are conducted on each time series to find the most
              probable (maximal value from a group of the tests) values of "D" and "d". 
              If "D" and/or "d" > 0, the seasonal differencing of order "D"
              and/or differencing of order "d" is applied to the time series.
   * Shift each time series if this time series has no values for the latest time points
       (NaNs rows at the end of time series) till these NaNs rows are gone
       for the month when the prediction must be done. If a shift is applied to the time series,
           the suffix "_lag{n}" is added to the time series name, where {n} is the number
           of the applied shift steps.
   * Lags of the corresponding CPI component (Y data) are also added to the X data. Lags numbers
        are defined by `Y_LAGS` in `RMB_CPI_nowcasting_challenge/settings.py`.
   * 12 monthly dummy variables are also added to the X data.
   * COVID event is also added to the X data as a spike in April 2020.

* X and Y data are split in a such way that the last two years of data are used
to validate models by calculating RMSE using walk-forward validation.
The oldest data are used for hyperparameters tuning of models.

* The models are created and optimized using hyperparameters tuning on the oldest data.
The used models are defined by `MODELS` in `RMB_CPI_nowcasting_challenge/settings.py`.
The hyperparameters for models optimization using grid search are defined for each model in `RMB_CPI_nowcasting_challenge/grid.py`.
The optimal models then are validated on the last two years of data by calculating RMSE using walk-forward validation.
Finally, each optimal model is used to predict the CPI component for the next month.

* The best model is constructed by selecting the optimal model with the smallest RMSE for each CPI component.
* The best max model is constructed using the following steps:
   1. Validation threshold is calculated as `(1. + RANGE) x smallest RMSE` from the previous step. `RANGE` is defined in `RMB_CPI_nowcasting_challenge/settings.py`.
   2. All optimized models with RMSE less than the validation threshold are selected.
   3. The model giving the highest predicted CPI among all previously selected models is selected as the best max model.

For example, we have the following CPS11000 (restaurants and hotel) November predictions for all models:
````
restaurantsand hotels   model   RMSE	CPI
CPS11000                svr     0.00909	114.7
CPS11000                cb      0.00928	115.0
CPS11000                xgb     0.00929	114.9
CPS11000                rf      0.00930	114.7
CPS11000                br      0.00935	114.8
CPS11000                lars    0.00935	114.9
CPS11000                lgbm	0.00937	114.9
CPS11000                lasso	0.00939	114.9
CPS11000                knn     0.00943	115.2
CPS11000                ab      0.00944	116.1
CPS11000                sarimax	0.00956	115.2
CPS11000                sgd     0.00966	115.0
CPS11000                ridge	0.00997	114.1
````

The model with the lowest RMSE is SVR. Therefore, the value predicted by this model 114.7 will be used 
as a prediction for this CPI component in the best model.

The validation threshold in this case will be (1. + 0.1) x 0.00909 = 0.01. 
All models have RMSE smaller than this threshold. Therefore, the maximal prediction among 
all these models 116.1 will be used as a prediction for this CPI component in the best max model.

