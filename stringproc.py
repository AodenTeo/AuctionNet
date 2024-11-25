import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
import csv
import pandas as pd
from transformers import BertTokenizer, BertModel
import logging

# function: tokenize
# ---------------------------------------
# Takes in a Python array of strings, and splits each string into an array with each word separate,
# and each word taken to lower case
#
# @param text Python array of strings to be tokenized
#
# @returns Array of tokenized strings (which are themselves arrays
def tokenize(text, max_length=30):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize and convert to IDs
    encoded = tokenizer(text, 
                        padding=True, 
                        truncation=True, 
                        max_length=max_length, 
                        return_tensors='pt', 
                        return_attention_mask=True)
    
    # Extract the tokenized inputs and attention mask
    input_ids = encoded['input_ids']  # Token IDs for each sentence
    attention_mask = encoded['attention_mask']  # Attention mask (1 for real tokens, 0 for padding)

    return input_ids, attention_mask

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

# function: text_to_tensor
# -------------------------------------
# Takes in a Python array of strings, and converts each string
# to a token_id vector of the same length, and converts the output
# to a PyTorch tensor to be passed into the model
#
# @param text Python array of strings to be converted into token vectors
#
# @param embeddings Dictionary which allows us to look up the vector space embedding of each word 
# 
# @param unknown Embedding that we use for unknown words, calculated as the average of all 
# the embeddings. Calculated separately and passed into the function for efficiency 
# 
# @returns PyTorch tensor to be passed into the model representing the text
def text_to_tensor(text, embeddings, embedding_dim, unknown):
    # Tokenize the text
    tokenized_text = tokenize(text)
    
    # Replace tokens with embeddings
    embedded_text = [[embeddings[token] if token in embeddings else unknown for token in sentence] for sentence in tokenized_text]
    
    # Determine the length of each sequence
    sequence_lengths = [len(sentence) for sentence in embedded_text]
    
    # Find the maximum sequence length
    max_length = max(sequence_lengths)
    
    # Pad each sentence with a vector of zeros
    zero_vector = [0.0] * embedding_dim
    padded_embedded_text = [
        sentence + [zero_vector] * (max_length - len(sentence)) for sentence in embedded_text
    ]
    
    # Convert to PyTorch tensors
    padded_tensor = torch.tensor(padded_embedded_text)
    lengths_tensor = torch.tensor(sequence_lengths)  
    return padded_tensor, lengths_tensor