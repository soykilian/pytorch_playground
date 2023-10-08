import torch
import torch.optim as optim
import torch.nn as nn


class ECE655LinRegression(nn.Module):
   def __init__(self):
      super().__init__()
      self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
      self.w = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

   def forward(self, x):
      return self.b + self.w * x


torch.manual_seed(42)
b=torch.randn(1,requires_grad=True, dtype=torch.float)
w=torch.randn(1,requires_grad=True, dtype=torch.float)

ECE655model = ECE655LinRegression()
optimizer=optim.SGD([b,w], lr=0.1)

print("ECE655 model parameters are:"); print(list(ECE655model.parameters()))

print("\nECE655 model state is:"); print(ECE655model.state_dict())

print("\nSGD optimizer state is:"); print(optimizer.state_dict())
