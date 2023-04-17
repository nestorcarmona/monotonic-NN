import torch
import torch.nn as nn
import math

class NonNegativeLinear(nn.Module):
    def __init__(self, in_features, out_features, nonnegative_bool, monotonic_bool, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nonnegative_bool = nonnegative_bool
        self.monotonic_bool = monotonic_bool

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        weight = self.weight.clone()
        for i, (non_negative, monotonic) in enumerate(zip(self.nonnegative_bool, self.monotonic_bool)):
            # input is monotonic, apply transformation
            if monotonic:
                # input is increasing, apply exp
                if non_negative is not None:
                    if non_negative:
                        # weight[:, i] = torch.pow(self.weight[:, i].clone(), 2) # torch.exp(weight[:, i])# self.weight[:, i].clone()
                        weight[:, i] = torch.exp(weight[:, i])
                        # weight[:, i].clamp_(min=0)

                    # input is decreasing, apply exp and negate
                    else:
                        # weight[:, i] = -torch.pow(self.weight[:, i].clone(), 2)# -torch.exp(weight[:, i]) # self.weight[:, i].clone()
                        weight[:, i] = -torch.exp(weight[:, i])
                        # weight[:, i].clamp_(max=0) 
        
        out = nn.functional.linear(input, weight, self.bias)
        # out = nn.functional.relu(out)
        return out
        # return nn.functional.linear(input, weight, self.bias)
        
class MinLayer(nn.Module):
    def __init__(self, num_groups, group_sizes):
        super().__init__()
        self.num_groups = num_groups
        self.group_sizes = group_sizes
        
    def forward(self, x):
        out = []
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
        out = x @ self.weight + self.bias
        out = torch.max(out, dim=1, keepdim=True)[0]
        return out

class MonotonicNet(nn.Module):
    def __init__(self, in_features, group_sizes, nonnegative_bool, monotonic_bool, n_hidden=1, hidden_size=[3]):
        super().__init__()
        self.num_groups_l1 = len(group_sizes[0])
        self.group_sizes_l1 = group_sizes[0]
        self.nonnegative_bool = nonnegative_bool
        self.monotonic_bool = monotonic_bool
        self.n_hidden = n_hidden
        # self.hidden_size = hidden_size if (n_hidden-1) > 0 else self.num_groups_l1

        self.layer1 = NonNegativeLinear(in_features=in_features, out_features=sum(self.group_sizes_l1), nonnegative_bool=nonnegative_bool, monotonic_bool=monotonic_bool, bias=True)
        self.min_layer = MinLayer(num_groups=self.num_groups_l1, group_sizes=self.group_sizes_l1)
        
        self.bn1 = nn.BatchNorm1d(self.num_groups_l1)

        # Create hidden linear layers and corresponding activation functions
        # self.hidden_layers = nn.ModuleList()
        # self.activation_functions = nn.ModuleList()
        
        n_features_in = self.num_groups_l1
        
        # for i in range(self.n_hidden - 1):
        #     self.hidden_layers.append(nn.Linear(in_features=n_features_in, out_features=hidden_size[i]))
        #     self.activation_functions.append(nn.ReLU())
        #     # update in and out features
        #     n_features_in = hidden_size[i]

        self.hidden_size = hidden_size[-1] if (self.n_hidden-1) > 0 else self.num_groups_l1
            
        self.max_layer = MaxLayer(in_features=self.hidden_size, out_features=1)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.min_layer(out)
        out = self.bn1(out)
        # Apply hidden linear layers and activation functions
        # for i in range(self.n_hidden - 1):
        #     out = self.hidden_layers[i](out)
        #     out = self.activation_functions[i](out)
        out = self.max_layer(out)

        return out
