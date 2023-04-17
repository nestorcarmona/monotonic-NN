import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# define a dense neural network with 2 hidden layers
class Net(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_size: list = [8, 8]) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], out_features)

        # batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size[0])
        self.bn2 = nn.BatchNorm1d(hidden_size[1])

    # define the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.sigmoid(self.fc1(x))
        # x = self.bn1(x)
        x = F.sigmoid(self.fc2(x))
        # x = self.bn2(x)
        x = self.fc3(x)
        return x
    
    # define the monotonicity loss: compute divergence of output w.r.t. input
    def monotonicity_loss(self, x: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        # print('loss')
        x.requires_grad_(True)
        y = self.forward(x)

        gradient = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, allow_unused=True)[0]
        # print("gradient shape: ", gradient.shape)
        
        selected_gradient = torch.mm(gradient, M)
        # print("selected gradient shape: ", selected_gradient.shape)
        directional_derivative = torch.sum(selected_gradient, dim=1)
        loss = torch.sum(torch.max(torch.zeros_like(directional_derivative), -directional_derivative))

        return loss