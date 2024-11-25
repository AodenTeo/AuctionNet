# File: scratch.py
# -----------------------------------------------------------
# Loads the dataset from the file
import torch
from torch.utils.data import Dataset
import pandas as pd
from stringproc import create_vocab_csv, text_to_tensor, tokenize
import numpy as np
from torchtext.data.utils import get_tokenizer

# function: load_glove_embeddings
# --------------------------------------------
# loads the Stanford NLP Glove embeddings at the speciied 
# file path with the specified dimension 
# 
# @returns Python object representing the glove embeddings loaded
# from the text file 
def load_glove_embeddings(filepath, embedding_dim):
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Load the downloaded Glove embeddings 
glove_path = "Glove/glove.6B.50d.txt"
embedding_dim = 50
glove_embeddings = load_glove_embeddings(glove_path, embedding_dim)

# standard embedding used for an unknown word. This is the average of all the embeddings 
# provided. We precompute this for efficiency.
UNKNOWN_EMBEDDING = [-0.12920076, -0.28866628, -0.01224866, -0.05676644, -0.20210965, -0.08389011,
 0.33359843, 0.16045167, 0.03867431, 0.17833012, 0.04696583, -0.00285802,
 0.29099807, 0.04613704, -0.20923874, -0.06613114, -0.06822549, 0.07665912,
 0.3134014, 0.17848536, -0.1225775, -0.09916984, -0.07495987, 0.06413227,
 0.14441176, 0.60894334, 0.17463093, 0.05335403, -0.01273871, 0.03474107,
 -0.8123879, -0.04688699, 0.20193407, 0.2031118, -0.03935686, 0.06967544,
 -0.01553638, -0.03405238, -0.06528071, 0.12250231, 0.13991883, -0.17446303,
 -0.08011883, 0.0849521, -0.01041659, -0.13705009, 0.20127155, 0.10069408,
 0.00653003, 0.01685157]

# Process the dataset from the file clean_art.csv
class ArtDataset(Dataset):
    def __init__(self):
        # # create the required vocabularies to represent the titles and the columns
        # vocab_artist = create_vocab_csv('clean_art.csv', "Artist")
        # vocab_title = create_vocab_csv('clean_art.csv', "Title")
        # self.vocab_artist = vocab_artist
        # self.vocab_title = vocab_title
        
        # read in the dataset as a pandas dataframe
        xy = pd.read_csv(
            'clean_art.csv',
            header=0,  # The first row contains column names
            quotechar='"',  # Use double quotes as the text qualifier
            skipinitialspace=True  # Skip spaces after delimiters
        )

        # extract the artist column
        artist_col = list(map(str, xy["Artist"].tolist()))
        self.artist, self.artist_seg_ids = tokenize(artist_col)
        print(f"Artist Indexed Tokens: {self.artist}\n Artist Segment IDs: {self.artist_seg_ids}")

        # extract the title column (and convert everything in the column to a string)
        title_col = list(map(str, xy["Title"].tolist()))
        self.title, self.title_seg_ids = tokenize(title_col)
        print(f"Title Indexed Tokens: {self.title}\n Title Segment IDs: {self.title_seg_ids}")
        
        # extract the columns from year to Gini coefficient
        numerics = xy.iloc[:, 2:13]

        # normalize all the numerical data to have mean 0 and standard deviation 1
        numerics_tensor = torch.tensor(numerics.values, dtype=torch.float32)
        self.numerics_mean = numerics_tensor.mean(dim=0)  # store the mean and std for use at test time
        self.numerics_std = numerics_tensor.std(dim=0)  # store the mean and std for use at test time
        self.numerics = (numerics_tensor - self.numerics_mean) / self.numerics_std
        
        # extract the prices (target) from the dataframe, and normalize it
        price_tensor = torch.tensor(list(map(float, xy["Real Price USD"].tolist()))).view(-1, 1)
        self.price_median = price_tensor.median(dim=0)[0]  # normalize to median to adjust for the massive expensive outliers 
        self.price_std = price_tensor.std(dim=0)
        self.price = (price_tensor - self.price_median) / self.price_std
        print("Dataset loaded successfully!")

    def __getitem__(self, index):
        return self.artist[index], self.artist_seg_ids[index], self.price[index]

    def __len__(self):
        return self.title.shape[0]
    
    def __getstring__(self, index): 
        return self.artist_string[index], self.title_string[index]