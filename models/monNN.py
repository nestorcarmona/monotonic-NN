import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Union

class MonotonicGroup(nn.Module):
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            increasing: List[bool]) -> None:
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.increasing = increasing

        stdv = 1. / math.sqrt(in_features)
        self.latent_weights = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-stdv, stdv))
        self.latent_bias = nn.Parameter(torch.Tensor(out_features).uniform_(-stdv, stdv))

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features)) # Monotonic(self.latent_weights)

        self.bias = nn.Parameter(torch.Tensor(out_features)) # Monotonic(self.latent_bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.clone()

        for i, incr in enumerate(self.increasing):
            if incr:
                weight[:, i] = torch.exp(weight[:, i])
            else:
                weight[:, i] = - torch.exp(weight[:, i])

        return nn.functional.linear(x, weight, self.bias)

class MonotonicLinear(nn.Module):
    def __init__(self, groups: List[MonotonicGroup]) -> None:
        super().__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = [g.forward(x) for g in self.groups]
        return torch.cat(output, dim=0).reshape(len(self.groups), x.shape[0], self.groups[0].out_features)

class MaxLayer(nn.Module):
    def __init__(self, num_groups: int, group_sizes: List[int]) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.group_sizes = group_sizes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        # loop through each group
        for i in range(self.num_groups):
            group = x[:, :self.group_sizes[i]]
            out.append(torch.max(group, dim=1, keepdim=True)[0])
            x = x[:, self.group_sizes[i]:]
        out = torch.cat(out, dim=1)
        return out

class MinLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.min(x, dim=1, keepdim=True)[0]
        return out


class MonotonicNet(nn.Module):
    def __init__(
            self, 
            in_features: int, 
            group_sizes: List[int], 
            increasing: List[bool]) -> None:
        
        super().__init__()
        self.in_features = in_features
        self.num_groups = len(group_sizes)
        self.group_sizes = group_sizes
        self.increasing = increasing
        
        # Hidden layer: linear unit
        self.layer1 = MonotonicGroup(in_features=in_features, out_features=sum(group_sizes), increasing=increasing)

        # batch normalization
        # self.bn1 = nn.BatchNorm1d(sum(group_sizes))

        # Hidden layer: max layer
        self.max_layer = MaxLayer(num_groups=self.num_groups, group_sizes=self.group_sizes)
        
        # Output layer: min layer
        self.min_layer = MinLayer(in_features=self.num_groups, out_features=1)
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.layer1(x)
        # out = self.bn1(out)
        out = self.max_layer(out)
        out = self.min_layer(out)
        return out