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
    channel_orders (list) : volterra module channel orders
    lowrank_order (int) : lowrank order for every volterra module, initial value:8
    
    learning_rate (float) : learning rate, initial value:0.01
    weight_decay (float) : weight_decay, initial value:0.01


    '''
    def __init__(self, writer:SummaryWriter, num_training_steps, dataset_name, 
                 input_shape, mid_dim, 
                 channel_dims, channel_orders, lowrank_order = 8, 
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

        self.test_step_mseloss = []
        self.test_step_maeloss = []
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

        self.log('val_loss', mse_loss)
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
            data_full_pred = torch.cat(timeseries_data, dim=1)
            data_tensor = data_full_pred[:, -windows_size, :]
        
        result_tensor = data_full_pred[:,-steps,:]
        mse_loss = F.mse_loss(result_tensor, target_tensor)
        mae_loss = F.l1_loss(result_tensor, target_tensor)

        self.test_step_mseloss.append(mse_loss)
        self.test_step_mseloss.append(mae_loss)


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
    
    def on_test_end(self):
        curr_epoch_test_mse = torch.stack(self.test_step_mseloss).mean()
        curr_epoch_test_mae = torch.stack(self.test_step_maeloss).mean()

        self.writer.add_scalar(f'{self.dataset_name}/test/mse', curr_epoch_test_mse, self.current_epoch)
        self.writer.add_scalar(f'{self.dataset_name}/test/mae', curr_epoch_test_mae, self.current_epoch)
        
        print(f'test finished, mse:{curr_epoch_test_mse}, mae:{curr_epoch_test_mae}')
        
        self.test_step_maeloss.clear()
        self.test_step_mseloss.clear()
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
        self.writer.add_hparams(self.hparam_dict,metric_dict)

        self.writer.flush()
        self.writer.close()
 
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


if __name__ == '__main__':
    x = torch.randn([10, 96, 63])
    in_shape = [96, 63]
    model = volterraModule(in_shape, output_dim=3, channel_dims=[2,1], channel_orders=[1,2], lowrank_order=8)
    output = model(x)
    print(output.shape)
    pass