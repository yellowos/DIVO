import sys
import math
from typing import List

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from model.base_model import N_order_kronecker_expand, generate_mask_tensor

class pinModel(torch.nn.Module):
    def __init__(self, input_shape, output_dim, channel_orders:List[int], channel_dims:List[int], lowrank_order = 8):
        super(pinModel, self).__init__()
        assert len(channel_orders) == len(channel_dims), 'please ensure channel orders has the same length with channel dims'
        self.output_dim = output_dim
        
        self.multi_channel_volterra = volterraModule(input_shape, channel_orders, channel_dims, lowrank_order=lowrank_order)

        mid_dim = sum(channel_dims)
        self.drop_out = torch.nn.Dropout(0.5)
        self.layer_norm = torch.nn.LayerNorm(mid_dim)
        self.fc_layer = torch.nn.Linear(mid_dim, output_dim, bias=True)

    def forward(self, x:torch.Tensor):
        batch_size, _, _ = x.shape

        x = self.multi_channel_volterra(x)
        x = self.drop_out(x)
        
        output = self.layer_norm(x)
        output:torch.Tensor = self.fc_layer(output)
        return(output)

class volterraModule(torch.nn.Module):
    def __init__(self, in_shape, channel_orders, channel_dims, lowrank_order=8) -> None:
        super(volterraModule, self).__init__()
        self.channel_dims = channel_dims
        self.channel_orders = channel_orders

        order = max(channel_orders)

        for i in range(1, order+1):
            curr_order_mask = generate_mask_tensor(in_shape, i)
            self.register_buffer(f'mask_order_{i}', curr_order_mask)

        self.multi_channel_modules = torch.nn.ModuleList()

        for idx, mid_dim in enumerate(channel_dims):
            order = channel_orders[idx]
            hidden_dim = mid_dim
            n_order_modules_list = torch.nn.ModuleList()
            for i in range(1, order+1):
                n_order_modules_list.append(nOrderModule_lowrank(in_shape, lowrank_order, hidden_dim, i))
    
            self.multi_channel_modules.append(n_order_modules_list)
            
    def forward(self, x:torch.Tensor):
        batch_size, _, _ = x.shape
        output = []

        for j, dim in enumerate(self.channel_dims):
            order = self.channel_orders[j]

            for i in range(1, order+1): 
                x_expand = N_order_kronecker_expand(x, i)
                mask = getattr(self, f'mask_order_{i}').to(x_expand.device).repeat(batch_size, 1, 1)

                x_expand = x_expand.masked_fill(~mask, 0)

                if i == 1:
                    tmp_output = self.multi_channel_modules[j][i-1](x_expand)
                else:
                    tmp_output = tmp_output.add(self.multi_channel_modules[j][i-1](x_expand))

            output.append(tmp_output)
        output = torch.cat(output, dim=-1)
        output = output.unsqueeze(1)

        return output


class nOrderModule_lowrank(torch.nn.Module):
    def __init__(self, input_shape, lowrank_order, output_dim, order, high_order_dropout = 0.3) -> None:
        super(nOrderModule_lowrank, self).__init__()
        self.order = order
        input_length, input_width = input_shape
        expanded_length, expanded_width = [int(math.pow(input_length, order)), int(math.pow(input_width, order))]
        self.high_order_dropout = high_order_dropout

        if order == 1:
            self.order0_kernel_bias = torch.nn.Parameter(torch.randn([output_dim]))
            self.kernel_metrix = torch.nn.Parameter(torch.randn([expanded_length, expanded_width, output_dim]))
        else:
            expanded_length = int(math.pow(input_length, order))
            expanded_width = int(math.pow(input_width, order))

            self.U = torch.nn.Parameter(torch.randn(expanded_length, lowrank_order))
            self.V = torch.nn.Parameter(torch.randn(expanded_width, lowrank_order))
            self.core = torch.nn.Parameter(torch.randn(lowrank_order, lowrank_order, output_dim))

    def forward(self, x_after_expand: torch.Tensor) -> torch.Tensor:
        batch_size = x_after_expand.shape[0]

        if self.order == 1:
            expanded_parameter = self.kernel_metrix.repeat(batch_size, 1, 1, 1)
            result = torch.einsum('bij, bijk -> bk', x_after_expand, expanded_parameter) + self.order0_kernel_bias
        else:
            # high order forward with lowrank
            temp1 = torch.einsum('bij,ip->bpj', x_after_expand, self.U)
            temp2 = torch.einsum('bpj,jq->bpq', temp1, self.V)
            result = torch.einsum('bpq,pqk->bk', temp2, self.core)
        
        return result
 
