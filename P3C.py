import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from P3A import get_val_loader, get_train_loader
from LinRegression import LinRegression
from train_loop import Train
plt.rcParams['font.family'] = 'Times New Roman'

n_epochs=200
eta = 5e-2
train_loader = get_train_loader()
val_loader = get_val_loader()
lr_model = nn.Sequential(nn.Linear(3,1))
optimizerSDG=optim.SGD(lr_model.parameters(), lr=eta)     # using SGD optimizer
l1_loss=nn.L1Loss()
train_loss, val_loss = Train(train_loader, val_loader, lr_model, l1_loss, optimizerSDG, n_epochs)
plt.figure()
plt.plot(train_loss, marker='.')
plt.plot(val_loss, marker='.')
plt.xlabel("Nº of epoch", fontsize=11)
plt.ylabel("Loss value", fontsize=11)
plt.grid(color="grey", linewidth="0.8", linestyle="--")
plt.legend(["Train loss", "Val loss"], fontsize=11)
plt.title("L1 loss")
plt.savefig('./L1.png')

