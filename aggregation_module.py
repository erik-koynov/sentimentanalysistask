import torch.nn as nn
import torch

class AggregationModule(nn.Module):
    def __init__(self, n_classes = 4):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1,n_classes))
        self.b = nn.Parameter(torch.ones(1,n_classes))

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return self.a*a + self.b*b
