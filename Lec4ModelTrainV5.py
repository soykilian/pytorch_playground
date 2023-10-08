# Adapted from Dan Godoy Book, Chapter 2
# Model Training V5
# Modified by Dr. Tolga Soyata    9/4/2023

# Run this file after running Lec4DataPrepV2.py and Lec4ModelConfigV3.py
from Lec4DataPrepV2 import train_loader, val_loader
from Lec4ModelConfigV3 import device, train_step_fn, val_step_fn, writer


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


def mini_batch(device, data_loader, step_fn):
    mini_batch_losses = []
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        mini_batch_loss = step_fn(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)

    loss = np.mean(mini_batch_losses)
    return loss



n_epochs = 50
losses = []
val_losses = []
for epoch in range(n_epochs):
    loss = mini_batch(device, train_loader, train_step_fn)
    losses.append(loss)

    # VALIDATION - no gradients
    with torch.no_grad():
        val_loss = mini_batch(device, val_loader, val_step_fn)
        val_losses.append(val_loss)

    # Record train and vaidation losses for each epoch under tag "loss"
    writer.add_scalars(main_tag='loss',
                       tag_scalar_dict={
                           'training':loss,
                           'validation': val_loss},
                       global_step=epoch)
    
#Close the writer at the end
writer.close()
    