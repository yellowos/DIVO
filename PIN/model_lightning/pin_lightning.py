import sys
import math
from typing import List 

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
from transformers.optimization import get_cosine_schedule_with_warmup
import lightning as L

from model_lightning.base_model import N_order_kronecker_expand

class LitPinModule(L.LightningModule):
    def __init__(self, writer:SummaryWriter, dataset_name, input_shape, output_dim, channel_dims, channel_orders, num_training_steps, learning_rate = 0.01, weight_decay=0.01, lowrank=True, lowrank_order = 8, pred_steps=96):
        super(LitPinModule, self).__init__()
        self.writer = writer
        self.backward_steps = pred_steps

        self.dataset_name = dataset_name
        # self.order = order
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_training_steps = num_training_steps

        self.validation_step_mseloss = []
        self.validation_step_maeloss = []

        self.train_step_mseloss = []
        self.train_step_maeloss = []
        self.on_validation_end_called_times = 0
        self.sota_mse = 10
        self.sota_mae = 10
        self.channel_orders = channel_orders

        self.drop_out = torch.nn.Dropout(0.5)
        self.multi_channel_modules = torch.nn.ModuleList()
        for mid_dim, order in zip(channel_dims, channel_orders):
            hidden_dim = mid_dim*pred_steps
            n_order_modules_list = torch.nn.ModuleList()
            if lowrank:
                for i in range(order+1):
                    n_order_modules_list.append(nOrderModule_lowrank(input_shape, lowrank_order, hidden_dim, i))
            else:
                for i in range(order+1):
                    n_order_modules_list.append(nOrderModule(input_shape, hidden_dim, i))
            self.multi_channel_modules.append(n_order_modules_list)

        mid_dim_sum = np.sum(channel_dims)
        hidden_dim = mid_dim_sum*pred_steps
        self.layer_norm = torch.nn.LayerNorm(hidden_dim)

        self.fc_layer = torch.nn.Linear(mid_dim_sum, output_dim, bias=True)
        # self.fc_layer = torch.nn.Linear(hidden_dim, output_dim*pred_steps, bias=True)
        # self.fc_layer = kroneckerLiner([pred_steps, mid_dim], [pred_steps, output_dim])
    
    def forward(self, x:torch.Tensor):        
        batch_size, _, _ = x.shape
        output = []
        for j, order in enumerate(self.channel_orders):
            for i in range(order+1): 
                x_expand = x if i == 0 else N_order_kronecker_expand(x, i)
                # if i >= 2:
                #     x_expand = self.drop_out(x_expand)
                if i == 0:
                    tmp_output = self.multi_channel_modules[j][i](x_expand)
                else:
                    tmp_output = tmp_output.add(self.multi_channel_modules[j][i](x_expand))
            
            tmp_output = tmp_output.reshape(batch_size, self.backward_steps, -1)
            output.append(tmp_output)
        output = torch.cat(output, dim=-1)
        output = self.layer_norm(output.view(batch_size, -1)).view(batch_size, self.backward_steps, -1)
        
        output = self.drop_out(output)
        # output = self.kronecker_liner(output)
        output = self.fc_layer(output)
        # output = output.reshape([batch_size, self.backward_steps, self.output_dim])
        return output

    def configure_optimizers(self):
        print(self.weight_decay)
        myOptimizer = torch.optim.AdamW(self.parameters(), self.learning_rate, weight_decay=self.weight_decay)
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
        self.writer.add_hparams(self.hparam_dict,metric_dict)

        self.writer.flush()
        self.writer.close()

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
    
class nOrderModule_lowrank(torch.nn.Module):
    def __init__(self, input_shape, lowrank_order, output_dim, order, high_order_dropout = 0.3) -> None:
        super(nOrderModule_lowrank, self).__init__()
        self.order = order
        input_length, input_width = input_shape
        expanded_length, expanded_width = [int(math.pow(input_length, order)), int(math.pow(input_width, order))]
        self.high_order_dropout = high_order_dropout

        if order == 0:
            # 对于 0 阶，直接定义一个参数向量，形状为 (output_dim,)
            self.kernel_metrix = torch.nn.Parameter(torch.randn([output_dim]))
        elif order == 1:
            self.kernel_metrix = torch.nn.Parameter(torch.randn([expanded_length, expanded_width, output_dim]))
        else:
            # 计算展开后矩阵的尺寸：input_length^order 和 input_width^order
            expanded_length = int(math.pow(input_length, order))
            expanded_width = int(math.pow(input_width, order))
            # 使用 Tucker 分解将原本形状为 (expanded_length, expanded_width, output_dim) 的参数拆分为：
            # U: (expanded_length, mid_dim)
            # V: (expanded_width, mid_dim)
            # core: (mid_dim, mid_dim, output_dim)
            self.U = torch.nn.Parameter(torch.randn(expanded_length, lowrank_order))
            self.V = torch.nn.Parameter(torch.randn(expanded_width, lowrank_order))
            self.core = torch.nn.Parameter(torch.randn(lowrank_order, lowrank_order, output_dim))

    def forward(self, x_after_expand: torch.Tensor) -> torch.Tensor:
        batch_size = x_after_expand.shape[0]

        if self.order == 0:
            # 对于 0 阶，直接将参数复制到 batch 上
            result = self.kernel_metrix.repeat(batch_size, 1)
        elif self.order == 1:
            expanded_parameter = self.kernel_metrix.repeat(batch_size, 1, 1, 1)
            result = torch.einsum('bij, bijk -> bk', x_after_expand, expanded_parameter)
        else:
            # x_after_expand 的形状为 (batch_size, expanded_length, expanded_width)
            # 通过以下三步进行 Tucker 分解的乘法：
            # 1. 与 U 相乘：在 expanded_length 维上做乘法，结果形状 (batch_size, mid_dim, expanded_width)
            temp1 = torch.einsum('bij,ip->bpj', x_after_expand, self.U)
            # 2. 与 V 相乘：在 expanded_width 维上做乘法，结果形状 (batch_size, mid_dim, mid_dim)
            temp2 = torch.einsum('bpj,jq->bpq', temp1, self.V)
            # 3. 与核张量相乘，融合两个 mid_dim 维度到 output_dim：结果形状 (batch_size, output_dim)
            result = torch.einsum('bpq,pqk->bk', temp2, self.core)
        return result

    
class kroneckerLiner(torch.nn.Module):
    def __init__(self, in_shape: list, out_shape: list, bias: bool = True, lowrank_order:int = 8):
        super(kroneckerLiner, self).__init__()
        in_steps, mid_dim = in_shape
        out_steps, out_dim = out_shape
        self.out_shape = out_shape
        self.in_steps = in_steps
        self.mid_dim = mid_dim
        self.U = torch.nn.Parameter(torch.randn(in_steps, lowrank_order))
        self.V = torch.nn.Parameter(torch.randn(mid_dim, lowrank_order))

        self.core = torch.nn.Parameter(torch.randn(lowrank_order, lowrank_order, out_dim*out_steps))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_steps*out_dim))

    def forward(self, x):
        batch, _ = x.shape
        x = x.reshape(batch, self.in_steps, self.mid_dim)

        temp1 = torch.einsum('bij,ip->bpj', x, self.U)
        temp2 = torch.einsum('bpj,jq->bpq', temp1, self.V)
        result = torch.einsum('bpq,pqk->bk', temp2, self.core) + self.bias
        result.reshape([batch] + self.out_shape)

        return result