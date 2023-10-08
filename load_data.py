import sklearn.datasets
import torch
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

diabetes = sklearn.datasets.load_diabetes()
print(type(diabetes))
X = diabetes['data']
Y = diabetes['target']
corr = np.corrcoef(np.transpose(X))
feature_names = diabetes.feature_names
plt.figure()
sns.heatmap(corr, xticklabels=feature_names, yticklabels=feature_names)
plt.savefig('./corr.png')
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)


class CustomDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def get_train_loader(batch_size=X.shape[0]):
    train_dataset = CustomDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def get_val_loader(batch_size=X.shape[0]):
    val_dataset = CustomDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return val_loader



