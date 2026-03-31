import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import copy
np.random.seed(2023)
torch.manual_seed(2023)
import os
device = ('cuda' if torch.cuda.is_available() else 'cpu')
save_dir = "D:/dl/kaggle/2026dian/result"
print(device)


def my_softmax(x, dim=-1):
    x_exp = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    x_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_sum

class MLP(nn.Module):

    def __init__(self,input_dim,hidden_dim,output_dim):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,output_dim)
        self.dropout=nn.Dropout(0.2)
    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = my_softmax(x,dim=1)
        return x


def build_model(config):
  return MLP(config['input_dim'],
             config['hidden_dim'],
             config['output_dim']).to(config['device'])
