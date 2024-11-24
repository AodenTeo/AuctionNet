# File: scratch.py
# -----------------------------------------------------------
# Loads the dataset from the file

import torch
from torch.utils.data import Dataset
import pandas as pd
from model import create_vocab_csv, text_to_tensor, tokenize
import numpy as np
from torchtext.data.utils import get_tokenizer

def lower_tokenizer(sentence): 
    tokenizer = get_tokenizer("basic_english")
    upper = tokenizer(sentence)
    return [str(string).lower() for string in upper]
# Dataset Hyperparameter
max_len = 30

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

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Load the downloaded Glove embeddings 
glove_path = "glove.6B.50d.txt"
embedding_dim = 50
glove_embeddings = load_glove_embeddings(glove_path, embedding_dim)
# Define function to calculate average cosine similarity
def avg_cosine_similarity_with_museum(sentence, glove_embeddings, target_word="museum"):
    # Get the embedding for the target word
    target_embedding = glove_embeddings.get(target_word)
    if target_embedding is None:
        return 0.0
    
    # Tokenize the sentence and calculate similarities
    tokens = lower_tokenizer(sentence)
    similarities = []
    for token in tokens:
        embedding = glove_embeddings.get(token)
        if embedding is not None:  # Skip tokens not in the GloVe vocabulary
            similarities.append(cosine_similarity(embedding, target_embedding))
    
    # Return the average similarity
    return np.mean(similarities) if similarities else 0.0

# Process the dataset from the file art.csv
class ArtDataset(Dataset):
    def __init__(self):
        # create the required vocabularies to represent the titles and the columns
        vocab_artist = create_vocab_csv('clean_art.csv', "Artist")
        vocab_title = create_vocab_csv('clean_art.csv', "Title")
        self.vocab_artist = vocab_artist
        self.vocab_title = vocab_title
        
        # read in the dataset as a pandas dataframe
        xy = pd.read_csv(
            'clean_art.csv',
            header=0,  # The first row contains column names
            quotechar='"',  # Use double quotes as the text qualifier
            skipinitialspace=True  # Skip spaces after delimiters
        )

        # extract the artist column
        artist_col = list(map(str, xy["Artist"].tolist()))
        
        # compute the cosine similarity between each word in the list
        # and  
        artist_col_avg_similarities = np.array([
            avg_cosine_similarity_with_museum(artist, glove_embeddings) for artist in artist_col
        ])
        artist_col_avg_similarities = artist_col_avg_similarities.reshape(-1, 1)
        self.artist_col_numerical = torch.from_numpy(artist_col_avg_similarities).float()
        self.artist_string = artist_col
        self.artist = text_to_tensor(artist_col, vocab_artist, max_len)

        # extract the title column (and convert everything in the column to a string)
        title_col = list(map(str, xy["Title"].tolist()))
        title_col_avg_similarities = np.array([
            avg_cosine_similarity_with_museum(title, glove_embeddings) for title in title_col
        ])
        title_col_avg_similarities = title_col_avg_similarities.reshape(-1, 1)
        self.title_numerical = torch.from_numpy(title_col_avg_similarities).float()
        self.title_string = title_col
        self.title = text_to_tensor(title_col, vocab_title, max_len)
        
        # extract the columns from year to Gini coefficient
        numerics = xy.iloc[:, 2:13]

        # normalize all the numerical data to have mean 0 and standard deviation 1
        numerics_tensor = torch.tensor(numerics.values, dtype=torch.float32)
        self.numerics_mean = numerics_tensor.mean(dim=0)  # store the mean and std for use at test time
        self.numerics_std = numerics_tensor.std(dim=0)  # store the mean and std for use at test time
        self.numerics = (numerics_tensor - self.numerics_mean) / self.numerics_std
        
        # Now, produce a single feature vector 
        self.x = torch.cat((self.artist_col_numerical, self.title_numerical, self.numerics), dim=1)
        # extract the prices (target) from the dataframe, and normalize it
        price_tensor = torch.tensor(list(map(float, xy["Real Price USD"].tolist()))).view(-1, 1)
        self.price_median = price_tensor.median(dim=0)[0]
        self.price_std = price_tensor.std(dim=0)
        self.price = (price_tensor - self.price_median) / self.price_std
        
        # set the vocabulary size for both the artist and the title 
        self.artist_vocab_len = len(vocab_artist)
        self.title_vocab_len = len(vocab_title)
        print("Dataset loaded successfully!")

    def __getitem__(self, index):
        return self.x[index], self.price[index]

    def __len__(self):
        return self.title.shape[0]
    
    def __getstring__(self, index): 
        return self.artist_string[index], self.title_string[index]