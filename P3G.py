import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from P3A import get_val_loader, get_train_loader
from LinRegression import LinRegression
import numpy as np
from train_loop import Train
import time
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
n_epochs = np.arange(20, 1000, 50)
train_loader = get_train_loader()
val_loader = get_val_loader()
SGD_times = []
Adam_times = []
train_SGD_losses = []
train_Adam_losses = []
l1_loss = nn.MSELoss()
for epoch in n_epochs:
    # SDG
    eta = 5e-2
    lr_model2 = nn.Sequential(nn.Linear(3, 1)).to(device)
    print("SDG")
    optimizerSDG = optim.SGD(lr_model2.parameters(), lr=eta)
    start = time.time()
    train_loss, val_loss = Train(train_loader, val_loader, lr_model2, l1_loss,
            optimizerSDG, epoch, device)
    SGD_times.append(time.time()-start)
    train_SGD_losses.append(train_loss[-1])
    # ADAM
    print("Adam")
    lr_model = nn.Sequential(nn.Linear(3, 1)).to(device)
    eta = 5e-2
    optimizerAdam = optim.Adam(lr_model.parameters(), lr=eta)
    start = time.time()
    train_loss, val_loss = Train(train_loader, val_loader, lr_model, l1_loss,optimizerAdam, epoch, device)
    Adam_times.append(time.time() - start)
    train_Adam_losses.append(train_loss[-1])
plt.figure()
plt.plot(n_epochs, train_Adam_losses)
plt.plot(n_epochs, Adam_times)
plt.plot(n_epochs, train_SGD_losses)
plt.plot(n_epochs, SGD_times)
plt.legend(["Adam: losses", "Adam:time", "SGD: losses", "SGD: times"])
plt.savefig('./partF_CPU.png')
