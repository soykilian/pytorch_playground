# Adapted from Dan Godoy Book, Chapter 2
# Modified by Dr. Tolga Soyata    9/3/2023

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


# Our Linear regression model
class ECE655LinRegression(nn.Module):
   def __init__(self):
      super().__init__()
      # Pick a random starting point for the b, w model parameters
      torch.manual_seed(42)
      self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
      self.w = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

   def forward(self, x):
      return self.b + self.w * x

N=100                          # N = number of data points
n_epochs=1000                  # number of epochs
true_b = 1
true_w = 2
eta=0.1           # Learning rate

#
# Generate artificial data: training and validation sets
#
x = np.random.rand(N, 1)
y = true_b + true_w * x + (.1 * np.random.randn(N, 1))

# Shuffles the indices
idx = np.arange(N)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:int(N*.8)]
# Uses the remaining indices for validation
val_idx = idx[int(N*.8):]

# Generates train and validation sets in numpy
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

# Converts numpy train, validation data to PyTorch tensors
x_train_tensor = torch.as_tensor(x_train).float()
y_train_tensor = torch.as_tensor(y_train).float()



ECE655model = ECE655LinRegression()    # .to(device) if you have a GPU
optimizer=optim.SGD(ECE655model.parameters(), lr=eta)     # using SGD optimizer
loss_fn=nn.MSELoss(reduction='mean')   # using MSE loss function

for epoch in range(n_epochs):
    ECE655model.train()                   # set model in training mode
    yhat = ECE655model(x_train_tensor)    # forward propogation (prediction)
    loss = loss_fn(yhat, y_train_tensor)  # compute the loss
    
    loss.backward()                       # back propogation
    
    optimizer.step()                      # update b, w using grads and lr
    optimizer.zero_grad()                 # we don't want to accumulate grad
    
    # Report the model state every 50 epochs
    if ((epoch % 50) == 0):    
        #print(f"Epoch {epoch:4}: ECE655 Model State:",end='')
        #print(list(ECE655model.parameters()))
        P=list(ECE655model.parameters())
        bpar=float(P[0].detach().numpy())
        wpar=float(P[1].detach().numpy())
        print(f"Epoch {epoch:4}:    ECE655 Model State:   ",end='')
        print(f"      b = {bpar:7.5f}         w = {wpar:7.5f}")
print("\n========================DONE====================:")

print(f"Epoch {epoch:4}: ECE655 Model State:",end='')
print(list(ECE655model.parameters()))

