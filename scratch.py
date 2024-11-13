# File: scratch.py
# -----------------------------------------------------------
# Loads the dataset from the file

import torch
from torch.utils.data import Dataset
import pandas as pd
from model import create_vocab_csv, text_to_tensor

# Dataset Hyperparameter
max_len = 30

# Process the dataset from the file art.csv
class ArtDataset(Dataset):
    def __init__(self):
        # create the required vocabularies to represent the titles and the columns
        vocab_artist = create_vocab_csv('clean_art.csv', "Artist")
        vocab_title = create_vocab_csv('clean_art.csv', "Title")
        
        # read in the dataset as a pandas dataframe
        xy = pd.read_csv(
            'clean_art.csv',
            header=0,  # The first row contains column names
            quotechar='"',  # Use double quotes as the text qualifier
            skipinitialspace=True  # Skip spaces after delimiters
        )

        # extract the artist column
        artist_col = list(map(str, xy["Artist"].tolist()))
        self.artist_string = artist_col
        self.artist = text_to_tensor(artist_col, vocab_artist, max_len)

        # extract the title column (and convert everything in the column to a string)
        title_col = list(map(str, xy["Title"].tolist()))
        self.title_string = title_col
        self.title = text_to_tensor(title_col, vocab_title, max_len)
        
        # extract the columns from year to Gini coefficient
        numerics = xy.iloc[:, 2:13]

        # normalize all the numerical data to have mean 0 and standard deviation 1
        numerics_tensor = torch.tensor(numerics.values, dtype=torch.float32)
        self.numerics_mean = numerics_tensor.mean(dim=0)  # store the mean and std for use at test time
        self.numerics_std = numerics_tensor.std(dim=0)  # store the mean and std for use at test time
        self.numerics = (numerics_tensor - self.numerics_mean) / self.numerics_std

        # extract the prices (target) from the dataframe, and normalize it
        price_tensor = torch.tensor(list(map(float, xy["Real Price USD"].tolist()))).view(-1, 1)
        self.price_median = price_tensor.median(dim=0)[0]
        print(f"Median Price: ${self.price_median.item():.2f}")
        print(f"Mean Price: ${price_tensor.mean(dim=0).item():.2f}")
        print("Normalized to MEDIAN")
        self.price_std = price_tensor.std(dim=0)
        self.price = (price_tensor - self.price_median) / self.price_std
        
        # set the vocabulary size for both the artist and the title 
        self.artist_vocab_len = len(vocab_artist)
        self.title_vocab_len = len(vocab_title)
        print("Dataset loaded successfully!")

    def __getitem__(self, index):
        return self.artist[index], self.title[index], self.numerics[index], self.price[index]

    def __len__(self):
        return self.title.shape[0]
    
    def __getstring__(self, index): 
        return self.artist_string[index], self.title_string[index]
dataset = ArtDataset()
