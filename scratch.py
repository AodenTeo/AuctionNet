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

        # Step 5: Reset index for cleanliness
        xy.reset_index(drop=True, inplace=True)

        # Step 6: Print the cleaned DataFrame
        pd.set_option('display.max_columns', None)
        print(xy.head())

        array = xy.to_numpy()
        print(array[0][0])

    def __getitem__(self, index):
        return index

    def __len__(self):
        return 1
dataset = ArtDataset()