import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from P3A import get_val_loader, get_train_loader
from LinRegression import LinRegression
from train_loop import Train
plt.rcParams['font.family'] = 'Times New Roman'

n_epochs = 100
eta = 5e-2
batch_size = 4
train_loader = get_train_loader(batch_size=batch_size)
val_loader = get_val_loader(batch_size=batch_size)
lr_model2 = nn.Sequential(nn.Linear(3,1))
optimizerSDG = optim.Adam(lr_model2.parameters(), lr=eta)
l1_loss = nn.MSELoss()
train_loss_4, val_loss_4 = Train(train_loader, val_loader, lr_model2, l1_loss, optimizerSDG, n_epochs)
plt.figure()
plt.plot(train_loss_4, marker='.')
plt.plot(val_loss_4, marker='.')
plt.xlabel("Nº of epoch", fontsize=11)
plt.ylabel("Loss value", fontsize=11)
plt.grid(color="grey", linewidth="0.8", linestyle="--")
plt.legend(["Train loss", "Val loss"], fontsize=11)
#plt.savefig('./parte_plot4.png')

batch_size = 8
train_loader = get_train_loader(batch_size=batch_size)
val_loader = get_val_loader(batch_size=batch_size)
lr_model8 = LinRegression(input_size = 3)
optimizerSDG = optim.Adam(lr_model8.parameters(), lr=eta)
l1_loss = nn.MSELoss()
train_loss_8, val_loss_8 = Train(train_loader, val_loader, lr_model8, l1_loss, optimizerSDG, n_epochs)
plt.figure()
plt.plot(train_loss_8, marker='.')
plt.plot(val_loss_8, marker='.')
plt.xlabel("Nº of epoch", fontsize=11)
plt.ylabel("Loss value", fontsize=11)
plt.grid(color="grey", linewidth="0.8", linestyle="--")
plt.legend(["Train loss", "Val loss"], fontsize=11)
#plt.savefig('./parte_plot8.png')

batch_size = 16
#ta = 1e-2
train_loader = get_train_loader(batch_size=batch_size)
val_loader = get_val_loader(batch_size=batch_size)
lr_model2 = LinRegression(input_size = 3)
optimizerSDG = optim.Adam(lr_model2.parameters(), lr=eta)
l1_loss = nn.MSELoss()
train_loss_16, val_loss_16 = Train(train_loader, val_loader, lr_model2, l1_loss, optimizerSDG, n_epochs)
plt.figure()
plt.plot(train_loss_16, marker='.')
plt.plot(val_loss_16, marker='.')
plt.xlabel("Nº of epoch", fontsize=11)
plt.ylabel("Loss value", fontsize=11)
plt.grid(color="grey", linewidth="0.8", linestyle="--")
plt.legend(["Train loss", "Val loss"], fontsize=11)
#plt.savefig('./parte_plot16.png')

batch_size = 32
#ta = 0.05
train_loader = get_train_loader(batch_size=batch_size)
val_loader = get_val_loader(batch_size=batch_size)
lr_model32 = LinRegression(input_size = 3)
optimizerSDG = optim.Adam(lr_model32.parameters(), lr=eta)
l1_loss = nn.MSELoss()
train_loss_32, val_loss_32 = Train(train_loader, val_loader, lr_model32, l1_loss, optimizerSDG, n_epochs)
plt.figure()
plt.plot(train_loss_32, marker='.')
plt.plot(val_loss_32, marker='.')
plt.xlabel("Nº of epoch", fontsize=11)
plt.ylabel("Loss value", fontsize=11)
plt.grid(color="grey", linewidth="0.8", linestyle="--")
plt.legend(["Train loss", "Val loss"], fontsize=11)
#plt.savefig('./parte_plot32.png')

batch_size = 64
#eta = 0.1
train_loader = get_train_loader(batch_size=batch_size)
val_loader = get_val_loader(batch_size=batch_size)
lr_model64 = LinRegression(input_size = 3)
optimizerSDG = optim.Adam(lr_model64.parameters(), lr=eta)
l1_loss = nn.MSELoss()
train_loss_64, val_loss_64 = Train(train_loader, val_loader, lr_model64, l1_loss, optimizerSDG, n_epochs)
plt.figure()
plt.plot(train_loss_64, marker='.')
plt.plot(val_loss_64, marker='.')
plt.xlabel("Nº of epoch", fontsize=11)
plt.ylabel("Loss value", fontsize=11)
plt.grid(color="grey", linewidth="0.8", linestyle="--")
plt.legend(["Train loss", "Val loss"], fontsize=11)
#plt.savefig('./parte_plot64.png')

plt.figure()
plt.bar(['4', '8', '16', '32', '64'], [train_loss_4[-1], train_loss_8[-1], train_loss_16[-1], train_loss_32[-1], train_loss_64[-1]], align='center', alpha=1)
plt.bar(['4', '8', '16', '32', '64'], [val_loss_4[-1], val_loss_8[-1], val_loss_16[-1], val_loss_32[-1], val_loss_64[-1]], align='center', alpha = 0.8)
plt.ylim(0.5, 0.8)
plt.legend(["Train loss", "Val loss"])
plt.savefig('./parte_bar_train.png')
