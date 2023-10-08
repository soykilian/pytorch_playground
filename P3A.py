from ucimlrepo import fetch_ucirepo
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import csv
wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features
y = wine_quality.data.targets
print(wine_quality.metadata)
print(wine_quality.variables)
X = X.dropna()
X = X[['fixed_acidity', 'density', 'alcohol']]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
sc.fit(X_train)
X_train = pd.DataFrame(sc.transform(X_train))
X_val = pd.DataFrame(sc.transform(X_val))
import matplotlib.pyplot as plt

plt.scatter
class CustomDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.targets = torch.tensor(targets.values, dtype=torch.float32)

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
