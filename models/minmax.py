import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Monotonic(nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.latent = latent

    def getweights(self):
        return torch.exp(self.latent)

class MonotonicGroup(nn.Module):
    def __init__(self, in_features, out_features, increasing):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.increasing = increasing

        stdv = 1. / math.sqrt(in_features)
        self.latent_weights = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-stdv, stdv))
        self.latent_bias = nn.Parameter(torch.Tensor(out_features).uniform_(-stdv, stdv))

        self.weights = Monotonic(self.latent_weights)

        self.bias = Monotonic(self.latent_bias)

    def forward(self, x):
        if self.increasing:
            return F.linear(x, self.weights.getweights(), self.bias.getweights())
        else:
            return F.linear(x, -self.weights.getweights(), -self.bias.getweights())


class MonotonicLinear(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        output = [g.forward(x) for g in self.groups]
        return torch.cat(output, dim=0).reshape(len(self.groups), x.shape[0], self.groups[0].out_features)

class MonotonicMax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cat(tuple(torch.max(i, dim=1)[0].unsqueeze(1) for i in x), dim=1)

class MonotonicMin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.min(x, dim=1)[0].unsqueeze(1)
    
class MonotonicInteraction(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.interaction_layer = nn.Linear(input_size, input_size, bias=False)

    def forward(self, x):
        x = self.interaction_layer(x)
        return x
    
class MonotonicNet(nn.Module):
    def __init__(self, input_size, num_groups, group_sizes, increasing):
        super().__init__()
        self.input_size = input_size
        self.num_groups = num_groups
        self.group_sizes = group_sizes
        self.increasing = increasing


        self.layers = nn.ModuleList()
        for i in range(input_size):
            groups = []
            for j in range(num_groups[i]):
                group = MonotonicGroup(1, group_sizes[i], increasing[i])
                groups.append(group)

            self.layers.append(MonotonicLinear(nn.ModuleList(groups)))

        # in multimensional, take into account the interaction between variables
        if input_size > 1:
            self.interaction_layer = nn.Linear(sum(group_sizes), sum(group_sizes))

        self.m2 = MonotonicMax()
        self.m3 = MonotonicMin()

    def forward(self, x):
        x1 = []
        for i, layer in enumerate(self.layers):
            x1.append(layer(x[:, i].unsqueeze(1)))

        x = torch.cat(x1, dim=2)

        # in multimensional, take into account the interaction between variables
        if self.input_size > 1:
            x = self.interaction_layer(x)

        x = self.m2(x)
        x = self.m3(x)
        
        return x