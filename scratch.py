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
        xy = pd.read_csv('art.csv')

        array = xy.to_numpy()
        print(array[0][0])

    def __getitem__(self, index):
        return index

    def __len__(self):
        return 1
dataset = ArtDataset()
