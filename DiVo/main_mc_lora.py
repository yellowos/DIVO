from argparse import ArgumentParser
import random
import datetime
import sys

import lightning as L
import torch
from torch.utils.tensorboard import SummaryWriter

from model.pin_mc_lora import pinModel
from expirement.exp import expMain
from loader.dataloader_with_test import make_daataset, load_dataset_file
from utils.tools import str2bool, str2int_list, str2channel_dim_list

def get_args():
    parser = ArgumentParser()
    # dataset args
    parser.add_argument("--csv_file_path", help='csv dataset file path', default='dataset/weather.csv')
    parser.add_argument("--start_index", default=1, type=int)
    parser.add_argument("--stop_index", default='end')
    
    parser.add_argument("--batch_size", default=50, type=int)
    parser.add_argument("--windows_size", default=96, type=int)
    parser.add_argument("--pred_steps", default=96, type=int)
    parser.add_argument("--split", default='[7,2,1]')

    # model core args
    parser.add_argument("--multi_channel_orders", default='[1,2,2]')
    parser.add_argument("--multi_channel_dims", default='[d,h,h]')
    parser.add_argument("--lowrank_order", default=8, type=int)
    
    parser.add_argument("--learning_rate", default=0.008, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--max_epoch", default=50, type=int)
    
    # saver args
    parser.add_argument("--tag", default='timestamp')
    parser.add_argument("--logdir", default='auto')

    parser.add_argument("--seed", default='random')

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    args_dict = vars(args)

    max_epoch = args.max_epoch
    seed = random.randint(-0x8000000000000000, 0xffffffffffffffff) if args.seed == 'random' else int(args.seed)
    args_dict['seed'] = str(seed)
    torch.manual_seed(seed)

    device:str = args.device

    # prepare dataset
    csv_file_path:str = args.csv_file_path

    norm_df = load_dataset_file(csv_file_path, args.start_index, args.stop_index)
    feature_size = norm_df.shape[1]
    train_loader, val_loader, test_loader = make_daataset(norm_df, args.batch_size, args.windows_size, pred_length=args.pred_steps, split=str2int_list(args.split))

    # initialize model
    channel_dims = str2channel_dim_list(args.multi_channel_dims, feature_size)
    channel_orders = str2int_list(args.multi_channel_orders)
    model = pinModel([args.windows_size, feature_size], feature_size, channel_orders, channel_dims, lowrank_order=args.lowrank_order).to(device)
    
    # initialize SummaryWriter
    log_root_path = f"log/log_{device.replace(':', '_')}" if args.logdir == 'auto' else args.logdir
    run_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M") if args.tag == 'timestamp' else args.tag
    writer = SummaryWriter(f'{log_root_path}/tensorboard/{run_tag}')
    # writer.add_hparams(args_dict, {})

    # initialize expirement
    max_steps = len(train_loader)
    _, file = csv_file_path.rsplit('/', maxsplit=1)
    dataset_name = file.rstrip('.csv')
    my_exp = expMain(writer, args_dict, dataset_name, device,  
                     model, train_loader, val_loader, test_loader, max_epoch, max_steps, 
                     learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    
    my_exp.fit()
    my_exp.test()


if __name__ == '__main__':
    main()

