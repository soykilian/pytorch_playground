import torch
import torch.nn as nn

class LinRegression(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        # Pick a random starting point for the b, w model parameters
        torch.manual_seed(2022)
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.w = nn.Parameter(torch.randn((1,input_size), requires_grad=True, dtype=torch.float))

    def forward(self, x):
        weighted_sum = torch.matmul(x, self.w.t())
        return self.b + weighted_sum

