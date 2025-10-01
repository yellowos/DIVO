import sys
import math
from typing import List 

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
from transformers.optimization import get_cosine_schedule_with_warmup
import lightning as L

from model.base_model import N_order_kronecker_expand, generate_mask_tensor

class LitPinModule(L.LightningModule):
    r'''

    Pin Module implement in Lightning way, test model in multi step inference way  
    
    Initialize Args:
    ---
    writer (SummaryWriter) : SummaryWriter for save log
    num_training_steps (int) : training steps in single epoch
    dataset_name (str) : datasetname such as weather for save log

    input_shape (list) : input shape except batch_size, expected to be [step, feature]
    mid_dim (int) : feature linear output dim for every feature channel

    channel_dims (list) :  volterra module channel output dims
    channel_orders (list) : volterra module channel orders
    lowrank_order (int) : lowrank order for every volterra module, initial value:8
    
    learning_rate (float) : learning rate, initial value:0.01
    weight_decay (float) : weight_decay, initial value:0.01

    '''
    def __init__(self, input_shape, mid_dim, channel_dims, lowrank_order = 8):
        super(LitPinModule, self).__init__()

        self.drop_out = torch.nn.Dropout(0.5)

        steps, features = input_shape
        self.features = features
        new_in_shape = [steps, mid_dim]

        self.feature_linear_para = torch.nn.Parameter(torch.randn(features, features, mid_dim))
        self.feature_linear_bias = torch.nn.Parameter(torch.randn(features, 1, mid_dim))

        self.feature_channel_modules = torch.nn.ModuleList()
        
        for feature_idx in range(features):
            curr_feature_volterra = volterraModule(new_in_shape, channel_dims, lowrank_order)
            self.feature_channel_modules.append(curr_feature_volterra)

        mid_dim_sum = np.sum(channel_dims)
        hidden_dim = features*mid_dim_sum

        self.layer_norm = torch.nn.LayerNorm(hidden_dim)
        self.fc_layer = torch.nn.Linear(hidden_dim, features, bias=True)
    

    def forward(self, x:torch.Tensor):
        batch_size, _, _ = x.shape
        channels_output = []

        x = x.unsqueeze(1).repeat(1, self.features, 1, 1)
        x = torch.einsum('bcsf,cfm->bcsm', x, self.feature_linear_para) + self.feature_linear_bias

        for idx, curr_feature_volterra in enumerate(self.feature_channel_modules):
            curr_feature_x = x[:,idx,:,:]
            channels_output.append(curr_feature_volterra(curr_feature_x))
        
        output = torch.cat(channels_output, dim=-1).view(batch_size, -1)
        output = self.layer_norm(output).view(batch_size, 1, -1)
        output = self.drop_out(output)

        output = self.fc_layer(output)
        return output
    
class volterraModule(torch.nn.Module):
    def __init__(self, in_shape, channel_dims, lowrank_order=8) -> None:
        super(volterraModule, self).__init__()
        self.channel_dims = channel_dims
        
        order = len(channel_dims)

        for i in range(1, order+1):
            curr_order_mask = generate_mask_tensor(in_shape, i)
            self.register_buffer(f'mask_order_{i}', curr_order_mask)

        self.multi_channel_modules = torch.nn.ModuleList()

        for idx, mid_dim in enumerate(channel_dims):
            order = idx + 1
            hidden_dim = mid_dim
            n_order_modules_list = torch.nn.ModuleList()
            for i in range(1, order+1):
                n_order_modules_list.append(nOrderModule_lowrank(in_shape, lowrank_order, hidden_dim, i))
    
            self.multi_channel_modules.append(n_order_modules_list)
            
    def forward(self, x:torch.Tensor):
        batch_size, _, _ = x.shape
        output = []

        for j, dim in enumerate(self.channel_dims):
            order = j+1
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
