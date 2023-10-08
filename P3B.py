import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import torch
from train_loop import Train
from P3A import get_val_loader, get_train_loader
from LinRegression import LinRegression
plt.rcParams['font.family'] = 'Times New Roman'

n_epochs_b=200
eta_b = 1e-2
train_loader_b = get_train_loader()
val_loader_b = get_val_loader()
lr_model = nn.Sequential(nn.Linear(3,1))
optimizerSDG_b=optim.SGD(lr_model.parameters(), lr=eta_b)     # using SGD optimizer
mse=nn.MSELoss(reduction='mean')
train_loss, val_loss = Train(train_loader_b, val_loader_b, lr_model, mse, optimizerSDG_b, n_epochs_b)
plt.figure()
plt.plot(train_loss, marker='.')
plt.plot(val_loss, marker='.')
plt.xlabel("Nº of epoch", fontsize=11)
plt.ylabel("Loss value", fontsize=11)
plt.grid(color="grey", linewidth="0.8", linestyle="--")
plt.legend(["Train loss", "Val loss"], fontsize=11)
plt.title("SDG")
plt.savefig('./SGD.png')
