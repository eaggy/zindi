# -*- coding: utf-8 -*-
"""
The file contains codes and description of SARB data (X data).

Created on 26.09.2023

@author: ihar
"""

# data codes of x-data #
########################
DATA_CODES_1 = [
    "KBP1033M",  # South African Reserve Bank: Utilisation of cash reserves - deficit (contra account debits)
    "KBP1036M",  # South African Reserve Bank: Assets: Total liquidity provided
    "KBP1066M",  # Corporation for Public Deposits (CPD): Assets: Total assets
    "KBP1077M",  # Liabilities of banking institutions: Total deposits
    "KBP1110M",  # Assets of banking institutions: Credit card debtors
    "KBP1122M",  # Assets of banking institutions: Overdrafts and loans
    "KBP1132M",  # Assets of banking institutions : Total assets
    "KBP1150M",  # Banking institutions: Total deposits by residents
    "KBP1152M",  # Banking institutions: Total deposits by non-residents
    "KBP1261M",  # Total value of credit card purchases processed during the period (all institutions)
    "KBP1358M",  # Total assets of the monetary sector
    "KBP1367M",  # All monetary institutions: Net credit extended to the government sector
    "KBP1369M",  # All monetary institutions : Credit extended to the domestic private sector: Total loans and advances
    "KBP1371M",  # Monetary aggregates / Money supply: M1
    "KBP1373M",  # Monetary aggregates / Money supply: M2
    "KBP1374M",  # Monetary aggregates / Money supply: M3
    "KBP1375M",  # All monetary institutions: Assets securitised
    "KBP1403M",  # Prime overdraft rate
    "KBP1417M",  # 12-months' fixed deposits
    "KBP1444W",  # API Money market interest rates: Sabor - South African Benchmark Overnight Rate on deposits
    "KBP1446W",  # API Money market interest rates: Overnight Foreign Exchange (FX) rate (Weighted average on Forex forwards until 2007/06/15)
    "KBP1450W",  # API Johannesburg Interbank Average Rate (JIBAR/JIBA rate): 3 months
    "KBP1474M",  # Total gross new mortgage loans and re-advances
    "KBP1480M",  # Total mortgage loans outstanding
    "KBP1512M",  # Monetary sector assets: Total foreign assets
]
DATA_CODES_2 = [
    "KBP2000M",  # Yield on loan stock traded on the stock exchange: Government bonds - 0 to 3 years
    "KBP2003M",  # API Yield on loan stock traded on the stock exchange: Government bonds - 10 years and over
    "KBP2004M",  # Yield on loan stock traded on the stock exchange: Eskom bonds
    "KBP2013M",  # Secondary Market: BEASSA Government Bond index (GOVI)
    "KBP2025M",  # Secondary market - Stock exchange transactions: JSE Market Capitalisation at month end (Bonds)
    "KBP2039A",  # Secondary market - Stock exchange transactions: Total value (turnover) of shares traded on the JSE
    "KBP2040M",  # Secondary market - Stock exchange transactions - Total number of bond transactions
    "KBP2050M",  # Net purchases of shares by non-residents on the Johannesburg Stock Exchange (JSE)
    "KBP2051M",  # Net purchases of bonds by non-residents on the Bond Exchange of South Africa (BESA)
    "KBP2072A",  # Real estate: Transfer duties
    "KBP2550M",  # Purchases of shares by non-residents on the JSE
    "KBP2551M",  # Sales of shares by non-residents on the JSE
    "KBP2553M",  # Purchases of bonds by non-residents on the Bond Exchange of South Africa
    "KBP2554M"   # Sales of bonds by non-residents on the Bond Exchange of South Africa
]
DATA_CODES_4 = [
    "KBP4003M",  # National government financing: Change in cash balances
    "KBP4070M",  # Discount/Premium/Revaluation on Government bonds
    "KBP4071M",  # Total net financing of national government deficit
    "KBP4073M",  # Total national government debt: Marketable domestic debt - Treasury bills: Up to 91 days
    "KBP4074M",  # Total national government debt: Marketable domestic debt - Treasury bills: 128 days
    "KBP4075M",  # Total national government debt: Marketable domestic debt - Treasury bills: 273 days
    "KBP4076M",  # Total national government debt: Marketable domestic debt - Treasury bills: 364 days
    "KBP4086M",  # Total domestic bonds of national government
    "KBP4446M",  # Total marketable foreign debt of national government: Bonds
    "KBP4450M",  # Total non-marketable foreign debt of national government denominated in foreign currencies
    "KBP4595M",  # Total national government tax revenue (net)
    "KBP4597M"   # Revenue: Total national government revenue
]
DATA_CODES_5 = [
    "KBP5020M",  # Balance of Payments: Change in reserve assets
    "KBP5021M",  # Balance of Payments: Change in liabilities related to reserves
    "KBP5023M",  # Balance of Payments: Change in gross gold & other foreign reserves
    "KBP5270M",  # Reserve Bank gold reserves : Amount as at end of period
    "KBP5271M",  # Reserve Bank Special Drawing Rights reserves: Amount as at end of period
    "KBP5272M",  # Reserve Bank other foreign exchange reserves: Amount as at end of period
    "KBP5273M",  # Total gold and other foreign reserves of the Reserve Bank : Amount as at end of period
    "KBP5277M",  # International liquidity position of the Reserve Bank
    "KBP5283M",  # Balance of Payments: Net monetisation (+)/demonetisation (-) of gold
    "KBP5315M",  # API Foreign exchange rate : SA cent per ECU Middle rates (R1 = 100 cents)
    "KBP5317M",  # Foreign exchange rate : SA cent per SDR Middle rates (R1 = 100 cents)
    "KBP5319M",  # API Foreign exchange rate : SA cent per Japanese yen Middle rates (R1 = 100 cents)
    "KBP5323M",  # Foreign exchange rate: SA cent per China Yuan Middle rate
    "KBP5338M",  # API Foreign exchange rate : SA cent per UK pound Middle rates (R1 = 100 cents)
    "KBP5339M",  # API Foreign exchange rate : SA cent per USA dollar Middle rates (R1 = 100 cents)
    "KBP5346M",  # Platinum price in Rand
    "KBP5347M",  # Palladium price in Rand
    "KBP5348M",  # Coal price in Rand
    "KBP5349M",  # Brent crude oil price in Rand
    "KBP5356M",  # London gold price in rand
    "KBP5393M",  # Nominal effective exchange rate of the rand: Average for the period - 20 trading partners
    "KBP5395M",  # Real effective exchange rate of the rand: Average for the period - 20 trading partners - Trade in manufactured goods
    "KBP5453M",  # Average daily net turnover against the S.A. Rand: Spot transactions: Total
    "KBP5457M",  # Average daily net turnover against the S.A. Rand: Forward transactions: Total
    "KBP5461M",  # Average daily net turnover against the S.A. Rand: Swap transactions; Total
    "KBP5473M",  # Average daily net turnover against the S.A. Rand: Total transactions: Total
    "KBP5477M",  # Average daily net turnover in third currencies: Total transactions: Total
    "KBP5478M",  # Total average daily net turnover
    "KBP5806M",  # Gross gold and other foreign reserves
]
DATA_CODES_6 = [
    "KBP6200L",  # Net saving by households
    "KBP6286L",  # Ratio of gross savings to GDP
    "KBP6287L",  # Ratio of saving by households to disposable income of house- holds
    "KBP6630L",  # Gross value added at basic prices of primary sector (GDP)
    "KBP6631L",  # Gross value added at basic prices of agriculture, foresty and fishing (GDP)
    "KBP6632L",  # Gross value added at basic prices of mining and quarrying (GDP)
    "KBP6633L",  # Gross value added at basic prices of secondary sector (GDP)
    "KBP6634L",  # Gross value added at basic prices of manufacturing (GDP)
    "KBP6635L",  # Gross value added at basic prices of electricity, gas and water (GDP)
    "KBP6636L",  # Gross value added at basic prices of construction (contractors) (GDP)
    "KBP6637L",  # Gross value added at basic prices of tertiary sector (GDP)
    "KBP6638L",  # Gross value added at basic prices of wholesale and retail trade, catering and accommodation (GDP)
    "KBP6639L",  # Gross value added at basic prices of transport, storage and communication (GDP)
    "KBP6640L",  # Gross value added at basic prices of finance, insurance, real estate and business services (GDP)
    "KBP6642L",  # Gross value added at basic prices of community, social and personal services (GDP)
    "KBP6643L",  # Gross value added at basic prices of general government services (GDP)
    "KBP6645L",  # Gross value added at basic prices of all industries (GDP)
    "KBP6647L",  # Gross value added at basic prices of other community, social and personal services (GDP)
]
DATA_CODES_7 = [
    "KBP7019K",  # Official unemployment rate
    "KBP7060N",  # Indicators of real economic activity: Mining production: Gold
    "KBP7061N",  # Indicators of real economic activity: Mining production excluding gold
    "KBP7063T",  # Indicators of real economic activity: Building plans passed
    "KBP7064T",  # Indicators of real economic activity: Buildings completed
    "KBP7067N",  # Indicators of real economic activity: Trade: Number of new vehicles sold
    "KBP7068N",  # Indicators of real economic activity: Electric current generated
    "KBP7082T",  # Manufacturing: Orders and sales: Sales
    "KBP7085N",  # Manufacturing: Total volume of production (Manufacturing)
    "KBP7086T",  # Indicators of real economic activity: Trade: Retail sales
    "KBP7087T",  # Indicators of real economic activity: Trade: Wholesale sales
    "KBP7090N",  # Leading indicator of South Africa
    "KBP7091N",  # Coincident indicator of South Africa
    "KBP7092N",  # Lagging indicator of South Africa
    "KBP7095N",  # Leading indicator of all the main trading partner countries
    "KBP7098N",  # Coincident indicator of all the main trading partner countries
    "KBP7180N",  # Producer prices of domestic output: Agriculture
    "KBP7182N",  # Total producer prices of domestic output: Mining
    "KBP7183N",  # Total producer prices of domestic output: Electricity and water
    "KBP7184N",  # Total producer prices of domestic output: Intermediate manu- factured goods
    "KBP7185N",  # Producer prices of final manufactured goods: Food products, beverages and tobacco
    "KBP7186N",  # Producer prices of final manufactured goods: Textiles, clothing and footwear
    "KBP7188N",  # Producer prices of final manufactured goods: Coke, petroleum , chemical, rubber and plastic products
    "KBP7189N",  # Producer prices of final manufactured goods: Metal, machine- ry, equipment and computing equipment
    "KBP7191N",  # Producer prices of final manufactured goods: Transport equipment
    "KBP7192N",  # Total producer prices of final manufactured goods
    "KBP7193N",  # Producer prices of final manufactured goods: Paper & printed products
    "KBP7194N",  # Producer prices of final manufactured goods: Electrical machinery, communication and metering equipment
    "KBP7198M",  # PMI: Prices
    "KBP7201M",  # Shipping rates: Baltic Dry Index
    "KBP7203M",  # BER: Constraints on current manufacturing activities: shortage of raw materials
    "KBP7204M"   # BER survey : Manufacturing, stocks of finished goods / expected demand
]
DATA_CODES_9 = [
    "KBP9018K",  # Household: Saving, gross last data point 2022 Q3
]

data_codes = []
data_codes.extend(DATA_CODES_1)
data_codes.extend(DATA_CODES_2)
data_codes.extend(DATA_CODES_4)
data_codes.extend(DATA_CODES_5)
data_codes.extend(DATA_CODES_6)
data_codes.extend(DATA_CODES_7)
data_codes.extend(DATA_CODES_9)


API_MAPPING = {
    "KBP1444W": "MMSD719A",  # South African Benchmark Overnight rate on deposits (SABOR)
    "KBP1446W": "MMSD721A",  # Implied rate on one-day rand funding in the foreign exchange swap market (Overnight FX rate)
    "KBP1450W": "MMSD502A",  # 3-month JIBAR (Johannnesburg Interbank Average rate)
    "KBP2003M": "CMJD004A",  # Daily average yield on government bonds with an outstanding maturity of 10 years and longer
    "KBP4073": "MMSD402A",   # Treasury bills - 91 day (tender rates)
    "KBP4074": "MMSD403A",   # Treasury bills - 128 day (tender rates)
    "KBP4075": "MMSD404A",   # Treasury bills - 273 day (tender rates)
    "KBP4076": "MMSD406A",   # Treasury bills - 364 day (tender rates)
    "KBP5315M": "EXCZ002D",  # Rand per Euro
    "KBP5319M": "EXCZ120D",  # Rand per Japanese Yen
    "KBP5338M": "EXCZ001D",  # Rand per British Pound
    "KBP5339M": "EXCX135D",  # Rand per US Dollar
    "KBP5356M": "GDPL203D",  # London gold price in rand
}
