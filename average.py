# File: Average.py
# ----------------------------------------------------------------------------------------
# Calculates the average of all the embeddings in a given file. This average is then used 
# to compute the embedding for unknown words in the embedding matrix 

import numpy as np

GLOVE_FILE = 'Glove/glove.6B.50d.txt'

# Get number of vectors and hidden dim
with open(GLOVE_FILE, 'r') as f:
    for i, line in enumerate(f):
        pass
n_vec = i + 1
hidden_dim = len(line.split(' ')) - 1

vecs = np.zeros((n_vec, hidden_dim), dtype=np.float32)

with open(GLOVE_FILE, 'r') as f:
    for i, line in enumerate(f):
        vecs[i] = np.array([float(n) for n in line.split(' ')[1:]], dtype=np.float32)

average_vec = np.mean(vecs, axis=0)
print(average_vec)