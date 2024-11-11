import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import math
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd


# Process the dataset from the file art.csv
class ArtDataset(Dataset):
    def __init__(self):
        # data loading
        # Step 1: Load the CSV with proper settings
        xy = pd.read_csv(
            'art.csv',
            header=0,  # The first row contains column names
            quotechar='"',  # Use double quotes as the text qualifier
            skipinitialspace=True  # Skip spaces after delimiters
        )

        # Step 2: Remove rows containing "bottle" or "bottles" in any column
        xy = xy[~xy.apply(lambda row: row.astype(str).str.contains('bottle', case=False, na=False).any(), axis=1)]

        # Step 3: Drop rows where the 'Date' column is empty, NaN, or only spaces
        xy = xy[xy['Date'].str.strip().replace('', pd.NA).notna()]

        # Step 4: Drop rows where the 'Price' column is empty, NaN, or only spaces
        xy = xy[xy['Price'].str.strip().replace('', pd.NA).notna()]

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

        ########################
        # Step 6: Convert estimates and prices to nominal USD
        # Load the FRB_H10.csv exchange rates data
        frb = pd.read_csv(
            'FRB_H10.csv',
            header=5,  # The 5th row (index 4) contains the column names
            index_col=0,  # The first column should be the index (Time Period)
            quotechar='"',  # Use double quotes as the text qualifier
            skipinitialspace=True  # Skip spaces after delimiters
        )

        # Print to check the data and column names
        #print(frb.head())
        #print(frb.columns)

        # Extract currency exchange columns and units from the FRB dataset
        currency_columns = frb.columns[1:]  # All columns except the first one (Time Period)
        units = frb.iloc[1, 1:].values  # Extract units from the second row (skip the first column)
        #print(units)

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

        # Apply the function to create the new columns in the DataFrame
        xy = xy.apply(convert_to_usd, axis=1)

        ########################
        # Step 7 Add inflation to dataframe and produce 'real' value columns

        # Step 1: Load and transform the CPI-U data
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
        print(cpiu_long)

        # Add the 'CPI' column to 'xy' based on the 'End Date'
        xy['CPI'] = xy['End Date'].map(cpiu_long['CPI'])

        # Step 3: Calculate Real Prices
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
        # Step 8: Reset index for cleanliness
        xy.reset_index(drop=True, inplace=True)

        # Step 9: Print the cleaned DataFrame
        pd.set_option('display.max_columns', None)
        print(xy.head())

        # unique_prefixes = xy['Estimate'].dropna().str[:3].unique()
        # print("Unique prefixes in the 'Estimate' column:", unique_prefixes)

        array = xy.to_numpy()
        print(array[0][0])

    def __getitem__(self, index):
        return index

    def __len__(self):
        return 1


dataset = ArtDataset()
