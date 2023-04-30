import torch
import torch.nn as nn
import math
from torch import backends

from typing import List, Union, Type

# fix all seeds
torch.manual_seed(0)

class NonNegativeLinear(nn.Module):
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            nonnegative_bool: Union[List[bool], List[Union[bool, None]], List[None]], 
            monotonic_bool: List[bool], 
            bias: bool = True) -> None:
        
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.nonnegative_bool = nonnegative_bool
        self.monotonic_bool = monotonic_bool

        # weight vector of shape (total number of neurons, number of input features)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    # can be improved lol
    def reset_parameters(self) -> None:
        # initialization of weight vector
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    # forward method
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # clone to keep graph
        weight = self.weight.clone()

        # loop through each variable
        for i, (non_negative, monotonic) in enumerate(zip(self.nonnegative_bool, self.monotonic_bool)):
            # input is monotonic, apply transformation; if not monotonic, unconstrained
            if monotonic:
                # input is increasing, apply exp
                if non_negative is not None:
                    if non_negative:
                        weight[:, i] = torch.exp(weight[:, i])

                    # input is decreasing, apply exp and negate
                    else:
                        weight[:, i] = - torch.exp(weight[:, i])
        
        # self.weight.data = weight
        out = nn.functional.linear(input, weight, self.bias)
        return out
        
class MinLayer(nn.Module):
    def __init__(self, num_groups, group_sizes):
        super().__init__()
        self.num_groups = num_groups
        self.group_sizes = group_sizes
        
    def forward(self, x):
        out = []
        # loop through each group
        for i in range(self.num_groups):
            group = x[:, :self.group_sizes[i]]
            out.append(torch.min(group, dim=1, keepdim=True)[0])
            x = x[:, self.group_sizes[i]:]
        out = torch.cat(out, dim=1)
        return out
    
class MaxLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x):
        # maximum of the inputs coming from "MinLayer"
        out = torch.max(x, dim=1, keepdim=True)[0]
        return out

class MonotonicNet(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        group_sizes: List[int], 
        nonnegative_bool: Union[List[bool], List[Union[bool, None]], List[None]],# List[Union[bool, Type[None]]], 
        monotonic_bool: List[bool]) -> None:

        super().__init__()

        self.num_groups_l1 = len(group_sizes)
        self.group_sizes_l1 = group_sizes
        self.nonnegative_bool = nonnegative_bool
        self.monotonic_bool = monotonic_bool

        # Hidden layer: linear unit
        self.layer1 = NonNegativeLinear(in_features=in_features, out_features=sum(self.group_sizes_l1), nonnegative_bool=nonnegative_bool, monotonic_bool=monotonic_bool, bias=True)
        
        # normalize weights of layer1
        if False in monotonic_bool:
            self.layer1 = nn.utils.weight_norm(self.layer1) # type: ignore

        # Batch normalization to keep same scale
        # self.bn1 = nn.BatchNorm1d(sum(self.group_sizes_l1))

        # Hidden layer: min layer
        self.min_layer = MinLayer(num_groups=self.num_groups_l1, group_sizes=self.group_sizes_l1)
        
        # Output layer: max layer
        self.max_layer = MaxLayer(in_features=self.num_groups_l1, out_features=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer1(x)
        out = self.min_layer(out)
        out = self.max_layer(out)
        return out
