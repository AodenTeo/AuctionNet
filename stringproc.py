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
