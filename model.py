import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
import csv
import pandas as pd

# function: tokenize
# ---------------------------------------
# Takes in a Python array of strings, and splits each string into an array with each word separate,
# and each word taken to lower case
#
# @param text Python array of strings to be tokenized
#
# @returns Array of tokenized strings (which are themselves arrays
def tokenize(text):
    tokenizer = get_tokenizer("basic_english")
    tokens_array = [tokenizer(sentence) for sentence in text]
    return tokens_array

# function: create_vocab
# ---------------------------------------------------
# creates a vocabulary object out of a python array of
# strings.
#
# @param text Array of strings to extract the vocabulary from
#
# @returns torchtext vocabulary object representing the vocabulary
# present in the string
def create_vocab(text):
    tokenized_text = tokenize(text)
    counter = Counter([word for sentence in tokenized_text for word in sentence])
    vocab = Vocab(counter, min_freq=1)

    # Set the default index for unknown tokens (if the token hasn't appeared before, we automatically set it
    # to this index)
    vocab.unk_index = vocab['<unk>']
    return vocab

# function: create_vocab_csv
# ---------------------------------------------------
# creates a vocabulary object out a column of a csv file
#
# @param file Path to the CSV file that we want to extract
#
# @param column_name String representing the column name
# for which we want to form a vocabulary
#
# @returns torchtext vocabulary object representing the vocabulary
# present in the string
def create_vocab_csv(file_path, column_name):
    # Open the CSV file
    with open(file_path, 'r') as file:
        # Create a CSV reader object
        reader = csv.reader(file)

        # Extract the header row
        header = next(reader)

        # Find the index of the desired column
        column_index = header.index(column_name)

        # Extract the values of the desired column
        text = [row[column_index] for row in reader]

    tokenized_text = tokenize(text)
    counter = Counter([word for sentence in tokenized_text for word in sentence])
    vocab = Vocab(counter, min_freq=1)

    # Set the default index for unknown tokens (if the token hasn't appeared before, we automatically set it
    # to this index)
    vocab.unk_index = vocab['<unk>']
    return vocab
create_vocab_csv('art.csv', "Artist")

# function: text_to_tensor
# -------------------------------------
# Takes in a Python array of strings, and converts each string
# to a token_id vector of the same length, and converts the output
# to a PyTorch tensor to be passed into the model
#
# @param text Python array of strings to be converted into token vectors
#
# @param vocab Vocabulary object which represents the vocabulary used in the text
#
# @param max_len Positive integer representing the point at which we have to cut off each
# string
#
# @returns PyTorch tensor to be passed into the model representing the text
def text_to_tensor(text, vocab, max_len):
    tokenized_text = tokenize(text)
    token_ids = [[vocab[token] for token in sentence[:max_len]] + [0] * (max_len - len(sentence)) for sentence in tokenized_text]
    return torch.tensor(token_ids)

######################## START ################################
# Sample data
titles = ["Mona Lisa", "A Mona German metalware part canteen of flatware and cutlery with shaped stems"]
artists = ["Leonardo Da Vinci", "Frank, the Grump"]
numerical_features = torch.tensor([[12000, 1], [8500, 0]],
                                  dtype=torch.float32)  # e.g., price estimate and binary feature

# Tokenization: Splits each title up into separate words, and sets everything to lower case
tokenized_titles = tokenize(titles)
tokenized_artists = tokenize(artists)

# Builds a vocabulary out of all the words that appear in the training set
vocab_artists = create_vocab_csv("test.csv", "Artist")
vocab_titles = create_vocab_csv("test.csv", "Title")

# Convert tokens to indices and pad sequences
max_len = 20  # set a max length to handle long titles

xy = pd.read_csv(
    'test.csv',
    header=0,  # The first row contains column names
    quotechar='"',  # Use double quotes as the text qualifier
    skipinitialspace=True  # Skip spaces after delimiters
)

artist_tensor = text_to_tensor(artists, vocab_artists, max_len)
title_tensor = text_to_tensor(titles, vocab_titles, max_len)

# Define the model
class ArtPricePredictor(nn.Module):
    def __init__(self, vocab_size_artist, vocab_size_title, embedding_dim_artist, embedding_dim_title, numerical_features_dim):
        super(ArtPricePredictor, self).__init__()

        # Embedding layer for text data
        self.embedding_artist = nn.Embedding(vocab_size_artist, embedding_dim_artist)
        self.embedding_title = nn.Embedding(vocab_size_title, embedding_dim_title)
        self.embedding_dim_artist = embedding_dim_artist
        self.embedding_dim_title = embedding_dim_title

        # Fully connected layers for combined features
        self.fc1 = nn.Linear(embedding_dim_artist + embedding_dim_title + numerical_features_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output layer for regression task

    def forward(self, artist, title, numerical_data):
        # Pass text through embedding layer
        artist_embedded = self.embedding_artist(artist)
        title_embedded = self.embedding_title(title)

        # Pool the embeddings across the sequence (mean pooling here)
        artist_pooled = artist_embedded.mean(dim=1)  # Shape: (batch_size, embedding_dim_artist)
        title_embedded = title_embedded.mean(dim=1) # Shape: (batch_size, embedding_dim_title)
        # Concatenate text and numerical features
        combined_features = torch.cat((artist_pooled,title_embedded, numerical_data), dim=1)

        # Forward pass through fully connected layers
        x = torch.relu(self.fc1(combined_features))
        x = torch.relu(self.fc2(x))
        output = self.fc3(x)  # Final output for price prediction
        return output


# Hyperparameters
vocab_size_title = len(vocab_titles)
vocab_size_artist = len(vocab_artists)
embedding_dim_artist = 50
embedding_dim_title = 50
numerical_features_dim = numerical_features.shape[1]

# Instantiate model and print
model = ArtPricePredictor(vocab_size_artist, vocab_size_title, embedding_dim_artist, embedding_dim_title, numerical_features_dim)

# Forward Pass
output = model(artist_tensor, title_tensor, numerical_features)
