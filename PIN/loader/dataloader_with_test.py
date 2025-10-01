from typing import Tuple, List

from torch.utils.data import DataLoader
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from loader.dataset import pin_timeseries_dataset


def load_dataset_file(csv_file_path:str, start_index:int, stop_index:str) -> Tuple[pd.DataFrame, int]:
    '''
    # load source csv file and preprocess  

    Parameters
    ---
        csv_file_path : str, source csv file path  
        strat_index: int, start idnex for model, no index or frame or timestamp cols  
        stop_index : int, stop index for model, end of file colunb in normal  
    Return
    ---
        dataframe after preprocess  
    '''
    source_df = pd.read_csv(csv_file_path)
    stop_index = source_df.shape[1] if stop_index == 'end' else int(stop_index)
    filtered_df = source_df.iloc[:,start_index:stop_index]
    scaler = StandardScaler()
    norm_df = pd.DataFrame(scaler.fit_transform(filtered_df), columns=filtered_df.columns)

    norm_df.to_csv(csv_file_path.rstrip('.csv')+'_norm.csv')

    return norm_df

def make_daataset(norm_df:pd.DataFrame, batch_size:int, windows_size:int, pred_length:int=1, split:List[int]=[6,2,2]):
    '''
    # generate train and test dataloader  

    Parameters
    ---
        norm_df: Dataframe, source data(pd.Dataframe) to make dataset  
        batch_size: int, dataloader batch size  
        windows_size: int, input sequence length  
        pred_length: int, predict sequence length  
        split: List of int, train, val, test ratio in full dataset, default=[6,2,2]  
    Return
    ---
        train_loader: train dataset loader  
        val_loader: validation dataset loader
        test_loader: test dataset loader
    '''
    train, val, test = split
    test_ratio = test / (train + val + test)
    val_ratio = val / (train + val)

    train_val_df, test_df = train_test_split(norm_df, test_size=test_ratio, shuffle=False)
    train_df, val_df = train_test_split(train_val_df, test_size=val_ratio, shuffle=False)

    train_dataset = pin_timeseries_dataset(train_df, windows_size=windows_size, pred_length=1)
    val_dataset = pin_timeseries_dataset(val_df, windows_size=windows_size, pred_length=1)
    test_dataset = pin_timeseries_dataset(test_df, windows_size=windows_size, pred_length=pred_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=20
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=20
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=20
    )

    return train_loader, val_loader, test_loader
