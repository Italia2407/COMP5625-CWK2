import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
import torchvision.io as io
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#import pandas as pd
#import matplotlib.pyplot as plt
from collections import Counter
import os
import numpy as np

print('Hello')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)