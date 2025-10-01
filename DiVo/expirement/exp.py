from typing import List

from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.optim import SGD,AdamW,Adam
import torch
import torch.nn.functional as F
from tqdm import tqdm

class expMain:
    r'''

    model expirement  
    
    Initialize Args:
    ---
    writer (SummaryWriter) : SummaryWriter for save log,
    params_dict (dict) : args dict to save
    dataset_name (str) : datasetname such as weather for save log
    device (str) : device use

    model (torch.nn.Module): Pytorch model with basic forward
    train_loader (Dataloader) : train dataloader give data and target
    val_loader (Dataloader) : val dataloader give data and target
    test_loader (Dataloader) : test dataloader give data and target

    max_epoch (int) : max epoch in fit forcess
    max_step (int) : max steps in every train epoch 
    
    learning_rate (float) : learning rate, initial value:0.01
    weight_decay (float) : weight_decay, initial value:0.01

    '''
    def __init__(self, writer: SummaryWriter,params_dict:dict, dataset_name:str, device:str,
                 model: torch.nn.Module, train_loader, val_loader, test_loader,
                 max_epoch: int, max_steps: int, learning_rate=0.008, weight_decay=0.001):
        self.writer = writer
        self.params_dict = params_dict
        self.dataset_name = dataset_name
        self.device = device

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.max_epoch = max_epoch

        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(max_steps / 5),
            num_training_steps=max_steps,
            num_cycles=1
        )

    def fit(self):
        steps = 0
        for epoch in range(self.max_epoch):
            train_loop = tqdm(self.train_loader, desc=f"Epoch [{epoch + 1}/{self.max_epoch}] Training", leave=False, position=0)
            val_loop = tqdm(self.val_loader, desc=f"Epoch [{epoch + 1}/{self.max_epoch}] Validating", leave=False, position=0)

            total_train_mse = 0.0
            total_train_mae = 0.0
            train_steps = 0

            for step, batch in enumerate(train_loop):
                train_mse_loss, train_mae_loss = self._train_step(batch)
                train_mse_loss.backward()

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                total_train_mse += train_mse_loss.item()
                total_train_mae += train_mae_loss.item()
                train_steps += 1
                steps += 1

            avg_train_mse = total_train_mse / train_steps
            avg_train_mae = total_train_mae / train_steps

            total_val_mse = 0.0
            total_val_mae = 0.0
            val_steps = 0

            for step, batch in enumerate(val_loop):
                with torch.no_grad():
                    val_mse_loss, val_mae_loss = self._val_step(batch)
                    total_val_mse += val_mse_loss.item()
                    total_val_mae += val_mae_loss.item()
                    val_steps += 1

            avg_val_mse = total_val_mse / val_steps
            avg_val_mae = total_val_mae / val_steps

            print(f"--- Epoch [{epoch + 1}/{self.max_epoch}] - "
                f"Train MSE: {avg_train_mse:.4f}, Train MAE: {avg_train_mae:.4f} | "
                f"Val MSE: {avg_val_mse:.4f}, Val MAE: {avg_val_mae:.4f} ---")
            
            self.writer.add_scalar(f'{self.dataset_name}/train/mse', avg_train_mse, epoch)
            self.writer.add_scalar(f'{self.dataset_name}/train/mae', avg_train_mae, epoch)

            self.writer.add_scalar(f'{self.dataset_name}/val/mse', avg_val_mse, epoch)
            self.writer.add_scalar(f'{self.dataset_name}/val/mae', avg_val_mae, epoch)

            
    def test(self):
        print("--- Test Start ---")
        loop = tqdm(self.test_loader, desc="Testing", leave=True)

        total_test_mse = 0.0
        total_test_mae = 0.0
        test_steps = 0
        with torch.no_grad():
            for step, batch in enumerate(loop):
                test_mse_loss, test_mae_loss = self._test_step(batch)

                total_test_mse += test_mse_loss.item()
                total_test_mae += test_mae_loss.item()
                test_steps += 1
        
        avg_test_mse = total_test_mse / test_steps
        avg_test_mae = total_test_mae / test_steps

        self.writer.add_hparams(self.params_dict, {'test_mse':avg_test_mse, 'test_mae':avg_test_mae})
        print(f"--- Test MSE: {avg_test_mse:.4f}, Test MAE: {avg_test_mae:.4f} ---")


    def _train_step(self, batch):
        train_data, train_target = batch
        data_tensor:torch.Tensor = train_data.to(self.device)
        target_tensor:torch.Tensor = train_target.to(self.device)

        result_tensor:torch.Tensor = self.model.forward(data_tensor)

        mse_loss = F.mse_loss(result_tensor, target_tensor)
        mae_loss = F.l1_loss(result_tensor, target_tensor)

        return mse_loss, mae_loss

    def _val_step(self, batch):
        val_data, val_target = batch
        data_tensor:torch.Tensor = val_data.to(self.device)
        target_tensor:torch.Tensor = val_target.to(self.device)

        result_tensor:torch.Tensor = self.model.forward(data_tensor)
        
        mse_loss = F.mse_loss(result_tensor, target_tensor)
        mae_loss = F.l1_loss(result_tensor, target_tensor)

        return mse_loss, mae_loss

    def _test_step(self, batch):
        test_data, test_target = batch
        data_tensor:torch.Tensor = test_data.to(self.device)
        target_tensor:torch.Tensor = test_target.to(self.device)

        timeseries_data:List[torch.Tensor] = [data_tensor]
        ori_input = data_tensor
        
        _, steps, _ = target_tensor.shape
        _, windows_size, _ = data_tensor.shape

        for step in range(steps):
            result_tensor:torch.Tensor = self.model.forward(data_tensor)
            timeseries_data.append(result_tensor)
            data_full_pred = torch.cat(timeseries_data, dim=1)
            data_tensor = data_full_pred[:, -windows_size:, :]
        
        result_tensor = data_full_pred[:,-steps:,:]
        mse_loss = F.mse_loss(result_tensor, target_tensor)
        mae_loss = F.l1_loss(result_tensor, target_tensor)
        return mse_loss, mae_loss