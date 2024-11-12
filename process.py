# File: process.py
# ===============================================================
# Reads the raw data in final_prices.csv, and writes processed
# data to the clean_art.csv

import pandas as pd
import numpy as np
import sys

# Step 1: Load the CSV with proper settings
xy = pd.read_csv(
    'final_prices.csv',
    header=0,  # The first row contains column names
    quotechar='"',  # Use double quotes as the text qualifier
    skipinitialspace=True  # Skip spaces after delimiters
)

# Step 2: Remove rows containing "bottle" or "bottles" in any column
xy = xy[~xy.apply(lambda row: row.astype(str).str.contains('bottle', case=False, na=False).any(), axis=1)]
print('Length after removing wine: ', len(xy))
# Step 3: Drop rows where the 'Date' column is empty, NaN, or only spaces
xy = xy[xy['Date'].str.strip().replace('', pd.NA).notna()]
print('Length after removing no dates ', len(xy))

# Step 4: Drop rows where the 'Price' column is empty, NaN, or only spaces
xy = xy[xy['Price'].str.strip().replace('', pd.NA).notna()]
print('Length after removing no price ', len(xy))

# Step 5: Remove commas from the 'Estimate' and 'Price' columns
xy['Estimate'] = xy['Estimate'].str.replace(',', '', regex=False)
xy['Price'] = xy['Price'].str.replace(',', '', regex=False)

# Step 6: Extract the last 8 characters from the 'Date' column and reformat it as 'YYYY-MM'
month_mapping = {
    'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 'MAY': '05', 'JUN': '06',
    'JUL': '07', 'AUG': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
}

# Extract last 8 characters, format as 'YYYY-MM'
xy['End Date'] = xy['Date'].str[-8:].str.strip().apply(
    lambda x: f"{x[-4:]}-{month_mapping.get(x[:3], '00')}"
)
#
print(f'The entry that gets taken to NAN is:  {xy.iloc[749]}')
########################
# Step 6: Convert estimates and prices to nominal USD
# Load the FRB_H10.csv exchange rates data
frb = pd.read_csv(
    'FRB_H10_2.csv',
    header=5,  # The 5th row (index 4) contains the column names
    index_col=0,  # The first column should be the index (Time Period)
    quotechar='"',  # Use double quotes as the text qualifier
    skipinitialspace=True  # Skip spaces after delimiters
)

# Print to check the data and column names
# print(frb.head())
# print(frb.columns)

# Extract currency exchange columns and units from the FRB dataset
currency_columns = frb.columns[1:]  # All columns except the first one (Time Period)
units = frb.iloc[1, 1:].values  # Extract units from the second row (skip the first column)
# print(units)

# Define the list of currency codes in the correct order (from AUD to TWD)
currency_codes = [
    'AUD', 'EUR', 'NZD', 'GBP', 'DKK', 'HKD', 'INR', 'JPY', 'KRW',
    'NOK', 'SGD', 'CHF', 'TWD'
]

# Extract the currency prefix from the 'Price' column and clean it
xy['Currency'] = xy['Price'].str[:3]  # Get the first 3 characters as the currency prefix


def convert_to_usd(row):
    currency = row['Currency']
    end_date = row['End Date']

    # Handle USD directly: if currency is USD, no conversion is needed
    if currency == 'USD':
        # Remove 'USD' prefix and convert to float
        price = row['Price'].strip()
        price_usd = float(price[4:])  # Assume 'USD 667', so we take from index 4 onward
        row['Nominal Price USD'] = price_usd

        # Process estimate if it exists
        estimate = row['Estimate'].strip()
        if " - " in estimate:
            # Extract lower and upper bounds from the estimate range
            estimate_range = estimate[4:].split(" - ")
            lb = float(estimate_range[0])
            ub = float(estimate_range[1])
            row['Nominal LB Estimate USD'] = lb
            row['Nominal UB Estimate USD'] = ub
        else:
            # If there's no range, set LB and UB to the same value as the price
            row['Nominal LB Estimate USD'] = price_usd
            row['Nominal UB Estimate USD'] = price_usd

        return row

    # If currency is not in the list of supported currencies, return missing value
    if currency not in currency_codes:
        return pd.NA

    # Find the corresponding column index for the currency
    currency_index = currency_codes.index(currency)
    currency_column = frb.columns[currency_index]  # Skip the first column (Time Period)

    # Find the exchange rate for the given end date
    exchange_row = frb.loc[end_date]
    if exchange_row.empty:
        return pd.NA  # If no data for the given month/year, return missing value

    # Extract the exchange rate for the currency
    exchange_rate = exchange_row[currency_column]

    # Determine the conversion factor based on the currency
    if currency_index < 4:  # AUD, EUR, NZD, GBP
        conversion_factor = exchange_rate  # USD per foreign currency
    else:  # For the rest (DKK to TWD), it's foreign currency per USD
        conversion_factor = 1 / exchange_rate

    # Handle the Nominal Prices USD (single price, like 'GBP 667')
    price = row['Price'].strip()  # Clean the price string (e.g., "GBP 667")

    # Convert single price into USD
    price_usd = float(price[4:]) * conversion_factor  # Remove the currency prefix and convert
    row['Nominal Price USD'] = price_usd  # Add this to the 'Nominal Prices USD' column

    # Handle the Estimate range (like 'GBP 400 - 600')
    estimate = row['Estimate'].strip()  # Clean the estimate string (e.g., "GBP 400 - 600")

    if " - " in estimate:
        # Extract lower and upper bounds from the estimate range
        estimate_range = estimate[4:].split(" - ")  # Remove the currency prefix and split
        lb = float(estimate_range[0]) * conversion_factor  # Lower Bound (LB)
        ub = float(estimate_range[1]) * conversion_factor  # Upper Bound (UB)

        # Add the converted bounds to new columns
        row['Nominal LB Estimate USD'] = lb
        row['Nominal UB Estimate USD'] = ub
    else:
        # If there's no range, set the LB and UB to the same value
        row['Nominal LB Estimate USD'] = price_usd
        row['Nominal UB Estimate USD'] = price_usd

    return row


print(f'After conversion to USD: {xy.iloc[749]}')
# Drop rows where the year is 2024 or greater and the month is greater than 10
xy = xy[~((xy['End Date'].str[:4].astype(int) >= 2024) & (xy['End Date'].str[5:7].astype(int) > 10))]

# Apply the function to create the new columns in the DataFrame
xy = xy.apply(convert_to_usd, axis=1)

########################
# Step 7 Add inflation to dataframe and produce 'real' value columns

# Load and transform the CPI-U data
cpiu = pd.read_csv('cpiu.csv')

cpiu = cpiu.iloc[:, :-2]
# Melt the CPI data to transform it into a long format
cpiu_long = cpiu.melt(id_vars=['Year'],
                      var_name='Month',
                      value_name='CPI')

# Create a 'Year-Month' column with format 'YYYY-MM'
month_mapping = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
    'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}
cpiu_long['Month'] = cpiu_long['Month'].map(month_mapping)
cpiu_long['Year-Month'] = cpiu_long['Year'].astype(str) + '-' + cpiu_long['Month']

# Drop unnecessary columns and set 'Year-Month' as the index
cpiu_long = cpiu_long[['Year-Month', 'CPI']].set_index('Year-Month')

# Ensure CPI values are treated as floats
cpiu_long['CPI'] = pd.to_numeric(cpiu_long['CPI'], errors='coerce')

# Add 'Inflation' column to the xy DataFrame
xy['Inflation'] = None  # Initialize a new column

# Iterate through each row in 'xy' and calculate inflation
for index, row in xy.iterrows():
    current_date = row['End Date']  # Get the current end date from the row

    # Check if the current_date exists in cpiu_long index (Year-Month)
    if current_date in cpiu_long.index:
        current_cpi = cpiu_long.loc[current_date, 'CPI']

        # Find the CPI for the same month in the previous year
        previous_year_date = str(int(current_date[:4]) - 1) + current_date[4:]  # Subtract 1 from the year
        if previous_year_date in cpiu_long.index:
            previous_cpi = cpiu_long.loc[previous_year_date, 'CPI']

            # Ensure both CPI values are floats for arithmetic operations
            current_cpi = float(current_cpi)
            previous_cpi = float(previous_cpi)

            # Calculate the annual inflation rate
            inflation_rate = ((current_cpi - previous_cpi) / previous_cpi) * 100

            # Assign the calculated inflation rate to the 'Inflation' column
            xy.at[index, 'Inflation'] = inflation_rate
        else:
            # If no previous year CPI data is found (e.g., the first entry of the dataset), set to NaN or None
            xy.at[index, 'Inflation'] = None

#print(cpiu_long)

# Add the 'CPI' and 'Inflation' columns to 'xy' based on the 'End Date'
xy['CPI'] = xy['End Date'].map(cpiu_long['CPI'])


# Calculate Real Prices
def adjust_for_inflation(row):
    cpi = row['CPI']
    if pd.isna(cpi) or cpi == 0:  # If CPI is missing or zero, keep the original values
        return row

    # Use a reference CPI for base year (e.g., use the latest CPI available as the base)
    base_cpi = 100  # Use the last available CPI as the base reference
    cpi = pd.to_numeric(cpi, errors='coerce')

    row['Real Price USD'] = (row['Nominal Price USD'] * base_cpi) / cpi
    row['Real LB Estimate USD'] = (row['Nominal LB Estimate USD'] * base_cpi) / cpi
    row['Real UB Estimate USD'] = (row['Nominal UB Estimate USD'] * base_cpi) / cpi

    return row


# Apply the function to the DataFrame
xy = xy.apply(adjust_for_inflation, axis=1)

########################
# Step 8: Load and transform the fed funds rate data
fed_funds = pd.read_csv(
    'FEDFUNDS.csv',
    header=0,  # Ensure the first row contains column names
    quotechar='"',
    skipinitialspace=True
)

# Clean the 'DATE' column by removing the last three characters to get 'YYYY-MM' format
fed_funds['DATE'] = fed_funds['DATE'].str[:-3]

# Create a dictionary mapping 'DATE' to 'FEDFUNDS' values for quick lookup
fed_funds_dict = fed_funds.set_index('DATE')['FEDFUNDS'].to_dict()

# Add a 'Nominal Fed Funds' column to xy by mapping the 'End Date' to the corresponding 'FEDFUNDS' rate
xy['Nominal Fed Funds'] = xy['End Date'].map(fed_funds_dict)

# Create the 'Real Fed Funds' column by subtracting 'Inflation' from 'Nominal Fed Funds'
xy['Real Fed Funds'] = xy['Nominal Fed Funds'] - xy['Inflation']
########################
# Step 9: Load and transform the NASDAQ data (NASDAQ Composite Index)
# Index Feb 5, 1971=100,
# Not Seasonally Adjusted
nasdaq = pd.read_csv(
    'NASDAQCOM.csv',
    header=0,  # Ensure the first row contains column names
    quotechar='"',
    skipinitialspace=True
)
# Clean the 'DATE' column by removing the last three characters to get 'YYYY-MM' format
nasdaq['DATE'] = nasdaq['DATE'].str[:-3]

# Filter the NASDAQ data for the years 1982-1984 to calculate the base index
nasdaq['NASDAQCOM'] = pd.to_numeric(nasdaq['NASDAQCOM'], errors='coerce')
nasdaq_filtered = nasdaq[nasdaq['DATE'].str[:4].isin(['1982', '1983', '1984'])]

# Calculate the average NASDAQ value for 1982-1984
nasdaq_base_value = nasdaq_filtered['NASDAQCOM'].mean()

# Scale the NASDAQ values so that the average value from 1982-1984 becomes 100
nasdaq['Scaled NASDAQ'] = (nasdaq['NASDAQCOM'] / nasdaq_base_value) * 100

# Create a dictionary mapping 'DATE' to 'Scaled NASDAQ' values for quick lookup
nasdaq_dict = nasdaq.set_index('DATE')['Scaled NASDAQ'].to_dict()

# Add the 'Scaled NASDAQ' column to 'xy' by mapping the 'End Date' to the corresponding 'Scaled NASDAQ' value
xy['Scaled NASDAQ'] = xy['End Date'].map(nasdaq_dict)
########################
# Step 10: Load and transform the UMCSENT (University of Michigan: Consumer Sentiment) Data
# Index 1966:Q1=100,
umcsent = pd.read_csv(
    'UMCSENT.csv',
    header=0,  # Ensure the first row contains column names
    quotechar='"',
    skipinitialspace=True
)
# Clean the 'DATE' column by removing the last three characters to get 'YYYY-MM' format
umcsent['DATE'] = umcsent['DATE'].str[:-3]

# Filter the UMCSENT data for the years 1982-1984 to calculate the base index
umcsent['UMCSENT'] = pd.to_numeric(umcsent['UMCSENT'], errors='coerce')
umcsent_filtered = umcsent[umcsent['DATE'].str[:4].isin(['1982', '1983', '1984'])]

# Calculate the average UMCSENT value for 1982-1984
umcsent_base_value = umcsent_filtered['UMCSENT'].mean()

# Scale the UMCSENT values so that the average value from 1982-1984 becomes 100
umcsent['Scaled UMCSENT'] = (umcsent['UMCSENT'] / umcsent_base_value) * 100

# Create a dictionary mapping 'DATE' to 'Scaled UMCSENT' values for quick lookup
umcsent_dict = umcsent.set_index('DATE')['Scaled UMCSENT'].to_dict()

# Add the 'Scaled UMCSENT' column to 'xy' by mapping the 'End Date' to the corresponding 'Scaled UMCSENT' value
xy['Scaled UMCSENT'] = xy['End Date'].map(umcsent_dict)
########################
# Step 10: Load and transform the GDP index data
gdp = pd.read_csv(
    'NA000334Q.csv',
    header=0,  # Ensure the first row contains column names
    quotechar='"',
    skipinitialspace=True
)
# Clean the 'DATE' column by removing the last three characters to get 'YYYY-MM' format
gdp['DATE'] = gdp['DATE'].str[:-3]

# Ensure the GDP column is numeric, coercing errors to NaN
gdp['GDP'] = pd.to_numeric(gdp['NA000334Q_NBD19470101'], errors='coerce')

# Filter the GDP data for the years 1982-1984 to calculate the base index
gdp_filtered = gdp[gdp['DATE'].str[:4].isin(['1982', '1983', '1984'])]

# Calculate the average GDP value for 1982-1984
gdp_base_value = gdp_filtered['GDP'].mean()

# Scale the GDP values so that the average value from 1982-1984 becomes 100
gdp['Scaled GDP'] = (gdp['GDP'] / gdp_base_value) * 100

# Step 4: Group GDP by quarter and get the last GDP value for each quarter
gdp['Year'] = gdp['DATE'].str[:4]
gdp['Month'] = gdp['DATE'].str[5:7]

# Create a 'Quarter' column based on the 'Month' column (1-3 for Q1, 4-6 for Q2, etc.)
gdp['Quarter'] = gdp['Month'].apply(lambda x: 'Q1' if x in ['01', '02', '03'] else
('Q2' if x in ['04', '05', '06'] else
 ('Q3' if x in ['07', '08', '09'] else 'Q4')))

# Group the GDP data by year and quarter and get the last scaled GDP for each quarter
quarterly_gdp = gdp.groupby(['Year', 'Quarter'])['Scaled GDP'].last().to_dict()

# Step 5: Add 'GDP Index' to 'xy' DataFrame based on the 'End Date'
# Let's assume you have a dataframe `xy` that has an 'End Date' column in 'YYYY-MM-DD' format

xy['End Date'] = pd.to_datetime(xy['End Date'], errors='coerce')


# Function to get GDP index based on the 'End Date'
def get_gdp_index(end_date):
    # Extract year and quarter from the 'End Date'
    year = end_date.year
    quarter = f"Q{((end_date.month - 1) // 3) + 1}"

    # Get the GDP index from the quarterly GDP dictionary
    return quarterly_gdp.get((str(year), quarter), None)


# Apply the function to the 'End Date' column and create the 'GDP Index' column
xy['GDP Index'] = xy['End Date'].apply(get_gdp_index)

########################
# Step 11 add real GDP growth
# Calculate the 'Real GDP Growth Rate' for each row in the 'gdp' DataFrame
gdp['Real GDP Growth Rate'] = gdp['Scaled GDP'].diff() / gdp['Scaled GDP'].shift(1) * 100


# Function to get Real GDP Growth Rate based on 'End Date'
def get_real_gdp_growth(end_date):
    # Extract year and quarter from the 'End Date'
    year = end_date.year
    quarter = f"Q{((end_date.month - 1) // 3) + 1}"

    # Get the Real GDP Growth Rate from the gdp dataframe for that year and quarter
    real_gdp_growth_rate = gdp[(gdp['Year'] == str(year)) & (gdp['Quarter'] == quarter)]['Real GDP Growth Rate']

    # Return the Real GDP Growth Rate or None if no data is found
    if not real_gdp_growth_rate.empty:
        return real_gdp_growth_rate.iloc[0]
    return None


print(f'Before mapping the end date to the corresponding DGS10 value: {xy.iloc[749]}')
# Apply the function to get 'Real GDP Growth Rate' and add it to the 'xy' DataFrame
xy['Real GDP Growth Rate'] = xy['End Date'].apply(get_real_gdp_growth)
########################
# Step 12:  Load and transform the market yield on treasury securities data
securities = pd.read_csv(
    'DGS10.csv',
    header=0,  # Ensure the first row contains column names
    quotechar='"',
    skipinitialspace=True
)
# Clean the 'DATE' column by removing the last three characters to get 'YYYY-MM' format
securities['DATE'] = securities['DATE'].str[:-3]

# Ensure 'End Date' in xy is formatted as 'YYYY-MM' (str)
xy['End Date'] = pd.to_datetime(xy['End Date']).dt.strftime('%Y-%m')

# Ensure 'DATE' in securities is formatted as 'YYYY-MM' (str)
securities['DATE'] = pd.to_datetime(securities['DATE']).dt.strftime('%Y-%m')

# Create a dictionary for the date to DGS10 mapping (using string dates)
date_to_yield = pd.Series(securities['DGS10'].values, index=securities['DATE']).to_dict()
print(f'Before mapping the end date to the corresponding DGS10 value: {xy.iloc[749]}')
# Map the 'End Date' in xy to the corresponding 'DGS10' value using the dictionary
xy['Nominal Securities Yield'] = xy['End Date'].map(date_to_yield)

xy['Inflation'] = pd.to_numeric(xy['Inflation'], errors='coerce')
print(f'Before calculating real securities Yield: {xy.iloc[749]}')
# Calculate the 'Real Securities Yield' using the Fisher equation
xy['Real Securities Yield'] = ((1 + xy['Nominal Securities Yield']) / (1 + xy['Inflation'])) - 1

########################
# Step 13:  Load the Gini coefficient data
gini = pd.read_csv(
    'SIPOVGINIUSA.csv',
    header=0,  # Ensure the first row contains column names
    quotechar='"',
    skipinitialspace=True
)
# Clean the 'DATE' column by removing the last six characters to get 'YYYY' format
gini['DATE'] = gini['DATE'].str[:-6]

print(f'Before ensuring End Date is in YYYY format: {xy.iloc[749]}')
# Ensure 'End Date' in xy is in 'YYYY' format (extract the year)
xy['Year'] = xy['End Date'].str[:4]

# Create a dictionary for mapping years to Gini coefficient values
year_to_gini = pd.Series(gini['SIPOVGINIUSA'].values, index=gini['DATE']).to_dict()
print(f'Before adding the Gini Coefficient: {xy.iloc[749]}')
# Map the 'Year' in xy to the corresponding Gini coefficient value using the dictionary
xy['Gini Coefficient'] = xy['Year'].map(year_to_gini)
print(f'Before splitting the month and year: {xy.iloc[749]}')
########################
# Step 14: Split end date into month and year
# Create new 'Month' and 'Year' columns
xy['End Date'] = pd.to_datetime(xy['End Date'])
xy['Month'] = xy['End Date'].dt.month
xy['Year'] = xy['End Date'].dt.year

#  Reset index for cleanliness
xy.reset_index(drop=True, inplace=True)

# Print the cleaned DataFrame
pd.set_option('display.max_columns', None)
#print(xy.columns)
#print(xy.head())
print(xy.iloc[749])

########################
# Step 15: Select relevant columns for final output
clean_art = xy[['Artist', 'CPI', 'Inflation', 'Real LB Estimate USD', 'Real Price USD',
                'Real UB Estimate USD', 'Title', 'Real Fed Funds', 'Scaled NASDAQ',
                'Scaled UMCSENT', 'GDP Index', 'Real GDP Growth Rate',
                'Real Securities Yield', 'Year', 'Gini Coefficient', 'Month']].copy()

# Convert 'Artist' and 'Title' to strings
clean_art['Artist'] = clean_art['Artist'].astype(str)
clean_art['Title'] = clean_art['Title'].astype(str)

# Convert all other columns to floats
columns_to_convert = ['CPI', 'Inflation', 'Real LB Estimate USD', 'Real Price USD',
                      'Real UB Estimate USD', 'Real Fed Funds', 'Scaled NASDAQ',
                      'Scaled UMCSENT', 'GDP Index', 'Real GDP Growth Rate',
                      'Real Securities Yield', 'Year', 'Gini Coefficient', 'Month']

clean_art[columns_to_convert] = clean_art[columns_to_convert].astype(float)
# Replace occurrences of the string 'nan' with actual NaN values
#clean_art.replace('nan', np.nan, inplace=True)

# Drop rows with any NaN values in any column
#clean_art = clean_art.dropna()
# (1) Remove any rows with missing entries in specified columns
clean_art.dropna(subset=['Artist', 'Title', 'Year', 'Month', 'CPI', 'Inflation',
                         'Real Fed Funds', 'Scaled NASDAQ', 'Scaled UMCSENT',
                         'GDP Index', 'Real GDP Growth Rate', 'Real Securities Yield',
                         'Gini Coefficient', 'Real LB Estimate USD',
                         'Real UB Estimate USD', 'Real Price USD'], inplace=True)

# (2) Reorder columns to match the desired order
clean_art = clean_art[['Artist', 'Title', 'Year', 'Month', 'CPI', 'Inflation',
                       'Real Fed Funds', 'Scaled NASDAQ', 'Scaled UMCSENT',
                       'GDP Index', 'Real GDP Growth Rate', 'Real Securities Yield',
                       'Gini Coefficient', 'Real LB Estimate USD', 'Real UB Estimate USD',
                       'Real Price USD']]

print(f'Number of cleaned art samples at the end: {len(clean_art)}')

# Write all data to a csv file
clean_art.to_csv('clean_art.csv', index=False)
