# Adapted from Dan Godoy Book, Chapter 2
# Model Configuration V3
# Modified by Dr. Tolga Soyata    9/4/2023

# We will run this file after running Lec4DataPrepV2.py
# Lec4DataPrepV2.py will compute train_loader
from Lec4DataPrepV2 import train_loader

import numpy as np
from sklearn.linear_model import LinearRegression

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def make_train_step_fn(model, loss_fn, optimizer):
     # Builds function that performs a step in the train loop
     def perform_train_step_fn(x, y):
         # Sets model to TRAIN mode
         model.train()
         # Step 1 - Computes model's predictions - forward pass
         yhat = model(x)
         # Step 2 - Computes the loss
         loss = loss_fn(yhat, y)
         # Step 3 - Computes gradients for "b" and "w" parameters
         loss.backward()
         # Step 4 - Updates parameters using gradients and
         # the learning rate
         optimizer.step()
         optimizer.zero_grad()
         # Returns the loss
         return loss.item()
     # Returns the function that will be called inside the train loop
     return perform_train_step_fn


def make_val_step_fn(model, loss_fn):
     # Builds function that performs a step in the validation loop
     def perform_val_step_fn(x, y):
         # Sets model to EVAL mode
         model.eval()
         # Step 1 - Computes model's predictions - forward pass
         yhat = model(x)
         # Step 2 - Computes the loss
         loss = loss_fn(yhat, y)
         # Do not calculate the gradients. Forward pass is all we need
         # Returns the loss
         return loss.item()
     # Returns the function that will be called inside the train loop
     return perform_val_step_fn


device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.1    # learning rate
torch.manual_seed(42)

####################### MODEL #########################
model=nn.Sequential(nn.Linear(1,1)).to(device)
# We are choosing the SGD optimizer
optimizer = optim.SGD(model.parameters(), lr=lr)
# We are choosing the MSE loss function
loss_fn=nn.MSELoss(reduction='mean')

# Creates the train_step function for this model and loss function
train_step_fn = make_train_step_fn(model, loss_fn, optimizer)
val_step_fn   = make_val_step_fn(model, loss_fn)


############### WRITER ##########
writer=SummaryWriter('runs/ECE655Lec4')

xdummy, ydummy = next(iter(train_loader))
writer.add_graph(model,xdummy.to(device))

