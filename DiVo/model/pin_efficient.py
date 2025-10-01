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

    Pin Module implement, test model in multi step inference way  
    
    Initialize Args:
    ---
    input_shape (list) : input shape except batch_size, expected to be [step, feature]
    mid_dim (int) : feature linear output dim for every feature channel

    channel_dims (list) :  volterra module channel output dims
    lowrank_order (int) : lowrank order for every volterra module, initial value:8

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
        
        order1_out_dim = sum(channel_dims)
        self.order1_para = torch.nn.Parameter(torch.randn([features, steps, mid_dim, order1_out_dim]))
        self.order1_bias = torch.nn.Parameter(torch.randn([features, order1_out_dim]))

        order2_mask = generate_mask_tensor(new_in_shape, 2)
        self.register_buffer('mask_order_2', order2_mask)
        order2_out_dim = channel_dims[1]
        self.order2_U = torch.nn.Parameter(torch.randn(features, int(math.pow(steps, 2)), lowrank_order))
        self.order2_V = torch.nn.Parameter(torch.randn(features, int(math.pow(mid_dim, 2)), lowrank_order))
        self.order2_core = torch.nn.Parameter(torch.randn(features, lowrank_order, lowrank_order, order2_out_dim))

        self.pad_dim = order1_out_dim-order2_out_dim
        mid_dim_sum = np.sum(channel_dims)
        hidden_dim = features*mid_dim_sum

        self.layer_norm = torch.nn.LayerNorm(hidden_dim)
        self.fc_layer = torch.nn.Linear(hidden_dim, features, bias=True)


    def forward(self, x:torch.Tensor):
        batch_size, step, feature = x.shape

        x = x.unsqueeze(1).repeat(1, self.features, 1, 1)
        x = torch.einsum('bcsf,cfm->bcsm', x, self.feature_linear_para) + self.feature_linear_bias

        x_reshape = x.reshape(batch_size*self.features, step, -1)
        x_expanded =  N_order_kronecker_expand(x_reshape, 2)
        mask = self.mask_order_2.to(x.device)
        x_expanded = x_expanded.masked_fill(~mask, 0)

        x_expanded = x_expanded.reshape(batch_size, self.features, x_expanded.shape[1], x_expanded.shape[2])

        order1_output = torch.einsum('bcsm,csmo->bco', x, self.order1_para) + self.order1_bias

        order2_temp1 = torch.einsum('bcsm,csp->bcpm', x_expanded, self.order2_U)
        order2_temp2 = torch.einsum('bcpm,cmq->bcpq', order2_temp1, self.order2_V)
        order2_output = torch.einsum('bcpq,cpqk->bck', order2_temp2, self.order2_core)
        padding_order2_output = F.pad(order2_output, (0, self.pad_dim), mode='constant', value=0)

        output = (order1_output + padding_order2_output).reshape(batch_size, -1)

        output = self.layer_norm(output).view(batch_size, 1, -1)
        output = self.drop_out(output)

        output = self.fc_layer(output)
        return output
    
