from argparse import ArgumentParser
import random
import datetime
import sys

import lightning as L
import torch
from torch.utils.tensorboard import SummaryWriter

from model_lightning.pin_lightning import LitPinModule, myCallBack
from loader.dataloader import make_daataset, load_dataset_file
from utils.tools import str2bool, str2index_list, str2channel_dim_list

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--csv_file_path", help='csv dataset file path', default='dataset/weather.csv')
    
    # parser.add_argument("--order", default=2, type=int)
    parser.add_argument("--multi_channel_orders", default='[1,2]')
    parser.add_argument("--multi_channel_dims", default='[d,h]')
    parser.add_argument("--pred_steps", default=96, type=int)

    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--windows_size", default=96, type=int)
    parser.add_argument("--lowrank_order", default=8, type=int)
    parser.add_argument("--lowrank", default='true')
    parser.add_argument("--devices", default='[0]')

    parser.add_argument("--start_index", default=1, type=int)
    parser.add_argument("--stop_index", default='end')
    
    parser.add_argument("--learning_rate", default=0.005, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)

    parser.add_argument("--max_epoch", default=50, type=int)

    parser.add_argument("--seed", default='random')
    parser.add_argument("--tag", default='timestamp')
    parser.add_argument("--logdir", default='auto')

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    max_epoch = args.max_epoch
    seed = random.randint(-0x8000000000000000, 0xffffffffffffffff) if args.seed == 'random' else int(args.seed)
    torch.manual_seed(seed)
    csv_file_path:str = args.csv_file_path

    lowrank = str2bool(args.lowrank)
    norm_df = load_dataset_file(csv_file_path, args.start_index, args.stop_index)
    feature_size = norm_df.shape[1]
    train_loader, test_loader = make_daataset(norm_df, args.batch_size, args.windows_size, pred_length=args.pred_steps)

    # mid_dim = feature_size*2 if args.mid_dim == 'double_feature' else int(args.mid_dim)
    channel_dims = str2channel_dim_list(args.multi_channel_dims, feature_size)
    channel_orders = str2index_list(args.multi_channel_orders)
    
    max_steps = len(train_loader)
    path, file = csv_file_path.rsplit('/', maxsplit=1)
    dataset_name = file.rstrip('.csv')
    devices_index_list = str2index_list(args.devices)
    
    log_root_path = f"log_cuda_{'_'.join(map(str, devices_index_list))}" if args.logdir == 'auto' else args.logdir

    run_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M") if args.tag == 'timestamp' else args.tag

    hparam_dict = {'lr':args.learning_rate, 'seed':str(seed), 'lowrank_order': args.lowrank_order, 
                    'pin_order':args.multi_channel_orders, 'batch_size':args.batch_size, 'windows_size':args.windows_size}
    
    writer = SummaryWriter(f'{log_root_path}/tensorboard/{run_tag}')

    # model = LitPinModule(writer, max_steps, dataset_name, [args.windows_size, feature_size], 10, 
    #                      channel_dims, channel_orders, learning_rate=args.learning_rate,
    #                      weight_decay=args.weight_decay, lowrank_order=args.lowrank_order)
    model = LitPinModule(writer, dataset_name, [args.windows_size, feature_size], feature_size, 
                         channel_dims, channel_orders, max_steps, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                         lowrank=lowrank, lowrank_order=args.lowrank_order, pred_steps=args.pred_steps)

    my_callback = myCallBack(writer, hparam_dict)
    trainer = L.Trainer(accelerator='gpu', devices=devices_index_list, max_epochs=max_epoch,
                        callbacks=[my_callback], default_root_dir=f'{log_root_path}/lightning/{run_tag}')
    
    trainer.fit(model, train_loader, test_loader)
    pass

if __name__ == '__main__':
    main()

