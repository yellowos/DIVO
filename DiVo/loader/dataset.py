from typing import Tuple
import os

from torch import Tensor
from torch.utils.data import Dataset
import torch
import pandas as pd

# from utils.type import 

class pin_timeseries_dataset(Dataset):
    def __init__(self, data_df:pd.DataFrame, windows_size:int, pred_length:int):
        super().__init__()
        self.source_data = data_df
        self.windows_size = windows_size
        self.pred_length = pred_length

    def __getitem__(self, index):
        data = self.source_data.iloc[index : index+self.windows_size, :]
        target = self.source_data.iloc[index+self.windows_size : index+self.windows_size+self.pred_length, :]
        data_tensor = torch.Tensor(data.values)
        target_tensor = torch.Tensor(target.values)

        return data_tensor, target_tensor
    
    def __len__(self):
        return len(self.source_data)-self.windows_size-self.pred_length

