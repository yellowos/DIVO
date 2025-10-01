import sys
import math
from typing import List 

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
from transformers.optimization import get_cosine_schedule_with_warmup
import lightning as L

from model_lightning.base_model import N_order_kronecker_expand, generate_mask_tensor

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
    lowrank_order (int) : lowrank order for every volterra module, initial value:8
    
    learning_rate (float) : learning rate, initial value:0.01
    weight_decay (float) : weight_decay, initial value:0.01


    '''
    def __init__(self, writer:SummaryWriter, num_training_steps, dataset_name, 
                 input_shape, mid_dim, channel_dims, lowrank_order = 8, 
                 learning_rate = 0.01, weight_decay=0.01):
        super(LitPinModule, self).__init__()

        self.writer = writer
        self.dataset_name = dataset_name
        self.num_training_steps = num_training_steps

        # self.order = order
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.validation_step_mseloss = []
        self.validation_step_maeloss = []

        self.train_step_mseloss = []
        self.train_step_maeloss = []
        self.on_validation_end_called_times = 0
        self.sota_mse = 10
        self.sota_mae = 10

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
        channels_output = []

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

    def configure_optimizers(self):
        print(self.weight_decay)
        myOptimizer = torch.optim.SGD(self.parameters(), self.learning_rate, weight_decay=self.weight_decay)
        lr_scheduler = get_cosine_schedule_with_warmup(myOptimizer, int(self.num_training_steps/5), self.num_training_steps)
        scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": True,
            "name": None
        }

        return {
            "optimizer":myOptimizer,
            "lr_scheduler":scheduler_config
        }
    
    def training_step(self, train_batch, batch_idx):
        train_data, train_target = train_batch
        data_tensor:torch.Tensor = train_data
        target_tensor:torch.Tensor = train_target

        result_tensor:torch.Tensor = self.forward(data_tensor)
        mse_loss = F.mse_loss(result_tensor, target_tensor)
        mae_loss = F.l1_loss(result_tensor, target_tensor)
        
        self.train_step_mseloss.append(mse_loss)
        self.train_step_maeloss.append(mae_loss)

        return mse_loss

    def validation_step(self, val_batch, batch_idx):
        val_data, val_target = val_batch
        data_tensor:torch.Tensor = val_data
        target_tensor:torch.Tensor = val_target

        result_tensor:torch.Tensor = self.forward(data_tensor)
        
        mse_loss = F.mse_loss(result_tensor, target_tensor)
        mae_loss = F.l1_loss(result_tensor, target_tensor)

        self.validation_step_mseloss.append(mse_loss)
        self.validation_step_maeloss.append(mae_loss)

        pass

    def test_step(self, batch, batch_idx):
        test_data, test_target = batch
        data_tensor:torch.Tensor = test_data
        target_tensor:torch.Tensor = test_target

        timeseries_data:List[torch.Tensor] = [data_tensor]
        ori_input = data_tensor
        
        _, steps, _ = target_tensor.shape
        _, windows_size, _ = data_tensor.shape

        for step in range(steps):
            result_tensor:torch.Tensor = self.forward(data_tensor)
            timeseries_data.append(result_tensor)
            data_tensor = torch.stack(timeseries_data, dim = 1)[:, -windows_size, :]


    def on_train_epoch_end(self):
        curr_epoch_train_mse = torch.stack(self.train_step_mseloss).mean()
        curr_epoch_train_mae = torch.stack(self.train_step_maeloss).mean()

        self.writer.add_scalar(f'{self.dataset_name}/train/mse', curr_epoch_train_mse, self.current_epoch)
        self.writer.add_scalar(f'{self.dataset_name}/train/mae', curr_epoch_train_mae, self.current_epoch)
        
        self.train_step_maeloss.clear()
        self.train_step_mseloss.clear()

    def on_validation_epoch_end(self):
        curr_epoch_val_mse = torch.stack(self.validation_step_mseloss).mean()
        curr_epoch_val_mae = torch.stack(self.validation_step_maeloss).mean()

        self.writer.add_scalar(f'{self.dataset_name}/val/mse', curr_epoch_val_mse, self.current_epoch)
        self.writer.add_scalar(f'{self.dataset_name}/val/mae', curr_epoch_val_mae, self.current_epoch)

        if curr_epoch_val_mse < self.sota_mse:
            self.sota_mse = curr_epoch_val_mse
            self.sota_mae = curr_epoch_val_mae
        
        self.log_dict({'sota_mse':self.sota_mse, 'sota_mae':self.sota_mae})

        self.validation_step_maeloss.clear()
        self.validation_step_mseloss.clear()
        pass

class myCallBack(L.Callback):
    def __init__(self, writer:SummaryWriter, hparam_dict:dict):
        self.writer = writer
        self.hparam_dict = hparam_dict

    def on_fit_end(self, trainer, pl_module):
        metric_dict = {'sota_mse':trainer.callback_metrics.get('sota_mse'), 
                       'sota_mae':trainer.callback_metrics.get('sota_mae')}
        self.writer.add_hparams(self.hparam_dict,metric_dict)

        self.writer.flush()
        self.writer.close()

    def on_exception(self, trainer, pl_module, exception):
        metric_dict = {'sota_mse':trainer.callback_metrics.get('sota_mse'), 
                       'sota_mae':trainer.callback_metrics.get('sota_mae')}
        self.writer.add_hparams(self.hparam_dict, metric_dict)

        self.writer.flush()
        self.writer.close()

# class volterraModule(torch.nn.Module):
#     def __init__(self, in_shape, channel_dims, lowrank_order=8) -> None:
#         super(volterraModule, self).__init__()
#         self.channel_dims = channel_dims
        
#         order = len(channel_dims)

#         for i in range(1, order+1):
#             curr_order_mask = generate_mask_tensor(in_shape, i)
#             self.register_buffer(f'mask_order_{i}', curr_order_mask)

#         self.multi_channel_modules = torch.nn.ModuleList()

#         for idx, mid_dim in enumerate(channel_dims):
#             order = idx + 1
#             hidden_dim = mid_dim
#             n_order_modules_list = torch.nn.ModuleList()
#             for i in range(1, order+1):
#                 n_order_modules_list.append(nOrderModule_lowrank(in_shape, lowrank_order, hidden_dim, i))
    
#             self.multi_channel_modules.append(n_order_modules_list)
            
#     def forward(self, x:torch.Tensor):
#         batch_size, _, _ = x.shape
#         output = []

#         for j, dim in enumerate(self.channel_dims):
#             order = j+1
#             for i in range(1, order+1): 
#                 x_expand = N_order_kronecker_expand(x, i)
#                 mask = getattr(self, f'mask_order_{i}').to(x_expand.device).repeat(batch_size, 1, 1)

#                 x_expand = x_expand.masked_fill(~mask, 0)

#                 if i == 1:
#                     tmp_output = self.multi_channel_modules[j][i-1](x_expand)
#                 else:
#                     tmp_output = tmp_output.add(self.multi_channel_modules[j][i-1](x_expand))

#             output.append(tmp_output)
#         output = torch.cat(output, dim=-1)
#         output = output.unsqueeze(1)

#         return output

# class nOrderModule_lowrank(torch.nn.Module):
#     def __init__(self, input_shape, lowrank_order, output_dim, order, high_order_dropout = 0.3) -> None:
#         super(nOrderModule_lowrank, self).__init__()
#         self.order = order
#         input_length, input_width = input_shape
#         expanded_length, expanded_width = [int(math.pow(input_length, order)), int(math.pow(input_width, order))]
#         self.high_order_dropout = high_order_dropout

#         if order == 1:
#             self.order0_kernel_bias = torch.nn.Parameter(torch.randn([output_dim]))
#             self.kernel_metrix = torch.nn.Parameter(torch.randn([expanded_length, expanded_width, output_dim]))
#         else:
#             expanded_length = int(math.pow(input_length, order))
#             expanded_width = int(math.pow(input_width, order))

#             self.U = torch.nn.Parameter(torch.randn(expanded_length, lowrank_order))
#             self.V = torch.nn.Parameter(torch.randn(expanded_width, lowrank_order))
#             self.core = torch.nn.Parameter(torch.randn(lowrank_order, lowrank_order, output_dim))

#     def forward(self, x_after_expand: torch.Tensor) -> torch.Tensor:
#         batch_size = x_after_expand.shape[0]

#         if self.order == 1:
#             expanded_parameter = self.kernel_metrix.repeat(batch_size, 1, 1, 1)
#             result = torch.einsum('bij, bijk -> bk', x_after_expand, expanded_parameter) + self.order0_kernel_bias
#         else:
#             # high order forward with lowrank
#             temp1 = torch.einsum('bij,ip->bpj', x_after_expand, self.U)
#             temp2 = torch.einsum('bpj,jq->bpq', temp1, self.V)
#             result = torch.einsum('bpq,pqk->bk', temp2, self.core)
        
#         return result

