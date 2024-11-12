import torch
from torch.utils.data import Dataset, DataLoader
import math
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
# Import the desired functions and classes from model.py
from model import create_vocab_csv, ArtPricePredictor, text_to_tensor

vocab_artist = create_vocab_csv('clean_art.csv', "Artist")
vocab_title = create_vocab_csv('clean_art.csv', "Title")

# HYPERPARAMETERS
max_len = 30

# Process the dataset from the file art.csv
class ArtDataset(Dataset):
    def __init__(self):
        # read in the dataset as a pandas dataframe
        xy = pd.read_csv(
            'clean_art.csv',
            header=0,  # The first row contains column names
            quotechar='"',  # Use double quotes as the text qualifier
            skipinitialspace=True  # Skip spaces after delimiters
        )

        # extract the artist column
        artist_col = list(map(str, xy["Artist"].tolist()))
        print(artist_col[0])
        self.artist = text_to_tensor(artist_col, vocab_artist, max_len)

        # extract the title column (and convert everything in the column to a string)
        title_col = list(map(str, xy["Title"].tolist()))
        print(title_col[0])
        self.title = text_to_tensor(title_col, vocab_title, max_len)
        print(self.artist[0], self.title[0])


    def __getitem__(self, index):
        return index

    def __len__(self):
        return 1


dataset = ArtDataset()
