import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# define a dense neural network with 2 hidden layers
class FCNet(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_hidden: list = [8, 8]) -> None:
        super(FCNet, self).__init__()

        self.net = []

        hs = [in_features] + n_hidden + [out_features]

        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend(
                [
                    nn.Linear(h0, h1),
                    nn.BatchNorm1d(h1),
                    nn.Sigmoid()
                ]
            )
        
        self.net.pop() # pop last batchnorm
        self.net.pop() # pop last activation function for output layer
        self.net = nn.Sequential(*self.net)

    # define the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) # type: ignore
    
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