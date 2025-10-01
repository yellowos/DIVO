import sys
import math

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from model.base_model import N_order_kronecker_expand

class pinModel(torch.nn.Module):
    def __init__(self, input_shape, output_dim, order, mid_dim):
        super(pinModel, self).__init__()
        self.order = order
        self.output_dim = output_dim
        
        self.n_order_modules_list = torch.nn.ModuleList()

        for i in range(order+1):
            self.n_order_modules_list.append(nOrderModule(input_shape, mid_dim, i))

        self.layer_norm = torch.nn.LayerNorm(mid_dim)
        self.fc_layer = torch.nn.Linear(mid_dim, output_dim, bias=True)

    def forward(self, x):
        for i in range(self.order+1):
            x_expand = N_order_kronecker_expand(x, i)
            if i == 0:
                output = self.n_order_modules_list[i](x_expand)
            else:
                output = output.add(self.n_order_modules_list[i](x_expand))
        
        output = self.layer_norm(output)
        output:torch.Tensor = self.fc_layer(output)
        output.unsqueeze_(1)
        return(output)

class nOrderModule(torch.nn.Module):
    def __init__(self, input_shape, output_dim, order) -> None:
        super(nOrderModule, self).__init__()
        self.order = order
        input_length, input_width = input_shape
        expanded_length, expanded_width = [int(math.pow(input_length, order)), int(math.pow(input_width, order))]
        
        if order == 0:
            self.kernel_metrix = torch.nn.Parameter(torch.randn([output_dim]))
        else:
            self.kernel_metrix = torch.nn.Parameter(torch.randn([expanded_length, expanded_width, output_dim]))

    def forward(self, x_after_expand:torch.Tensor) -> torch.Tensor:
        batch_size = x_after_expand.shape[0]

        if self.order == 0:
            result = self.kernel_metrix.repeat(batch_size, 1)
            pass
        else:
            expanded_parameter = self.kernel_metrix.repeat(batch_size, 1, 1, 1)
            result = torch.einsum('bij, bijk -> bk', x_after_expand, expanded_parameter)
        return result
 
