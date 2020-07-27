import argparse
import os
import torch
import time
from utils import seed_set, Option, toxcast_tasks
from model import TrimNet as Model
from trainer import Trainer
from dataset import load_dataset_scaffold, load_dataset_random, load_dataset_random_nan


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../data/', help="all data dir")
    parser.add_argument("--dataset", type=str, default='bace', help="muv,tox21,toxcast,sider,clintox,hiv,bace,bbbp")
    parser.add_argument('--seed', default=68, type=int)
    parser.add_argument("--gpu", type=int, nargs='+', default=0, help="CUDA device ids")

    parser.add_argument("--hid", type=int, default=32, help="hidden size of transformer model")
    parser.add_argument('--heads', default=4, type=int)
    parser.add_argument('--depth', default=3, type=int)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--batch_size", type=int, default=128, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument('--lr_scheduler_patience', default=10, type=int)
    parser.add_argument('--early_stop_patience', default=-1, type=int)
    parser.add_argument('--lr_decay', default=0.98, type=float)
    parser.add_argument('--focalloss', default=False, action="store_true")

    parser.add_argument('--eval', default=False, action="store_true")
    parser.add_argument("--exps_dir", default='../test', type=str, help="out dir")
    parser.add_argument('--exp_name', default=None, type=str)

    d = vars(parser.parse_args())
    args = Option(d)
    seed_set(args.seed)

    args.parallel = True if args.gpu and len(args.gpu) > 1 else False
    args.parallel_devices = args.gpu
    args.tag = time.strftime("%m-%d-%H-%M") if args.exp_name is None else args.exp_name
    args.exp_path = os.path.join(args.exps_dir, args.tag)

    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)
    args.code_file_path = os.path.abspath(__file__)

    if args.dataset == 'muv':
        args.tasks = ["MUV-466", "MUV-548", "MUV-600", "MUV-644", "MUV-652", "MUV-689", "MUV-692", "MUV-712", "MUV-713",
                      "MUV-733", "MUV-737", "MUV-810", "MUV-832", "MUV-846", "MUV-852", "MUV-858", "MUV-859"]
        args.out_dim = 2 * len(args.tasks)
        train_dataset, valid_dataset, test_dataset = load_dataset_random_nan(args.data, args.dataset, args.seed,
                                                                             args.tasks)
    elif args.dataset == 'tox21':
        args.tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
                      'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
        args.out_dim = 2 * len(args.tasks)
        train_dataset, valid_dataset, test_dataset = load_dataset_random(args.data, args.dataset, args.seed, args.tasks)
    elif args.dataset == 'toxcast':
        args.tasks = toxcast_tasks
        args.out_dim = 2 * len(args.tasks)
        train_dataset, valid_dataset, test_dataset = load_dataset_random_nan(args.data, args.dataset, args.seed,
                                                                             args.tasks)
    elif args.dataset == 'sider':
        args.tasks = ['SIDER1', 'SIDER2', 'SIDER3', 'SIDER4', 'SIDER5', 'SIDER6', 'SIDER7', 'SIDER8', 'SIDER9',
                      'SIDER10', 'SIDER11', 'SIDER12', 'SIDER13', 'SIDER14', 'SIDER15', 'SIDER16', 'SIDER17', 'SIDER18',
                      'SIDER19', 'SIDER20', 'SIDER21', 'SIDER22', 'SIDER23', 'SIDER24', 'SIDER25', 'SIDER26', 'SIDER27']
        args.out_dim = 2 * len(args.tasks)
        train_dataset, valid_dataset, test_dataset = load_dataset_random(args.data, args.dataset, args.seed, args.tasks)
    elif args.dataset == 'clintox':
        args.tasks = ['FDA_APPROVED']
        args.out_dim = 2 * len(args.tasks)
        train_dataset, valid_dataset, test_dataset = load_dataset_random(args.data, args.dataset, args.seed, args.tasks)
    elif args.dataset == 'hiv':
        args.tasks = ['HIV_active']
        train_dataset, valid_dataset, test_dataset = load_dataset_scaffold(args.data, args.dataset, args.seed,
                                                                           args.tasks)
        args.out_dim = 2 * len(args.tasks)
    elif args.dataset == 'bace':
        args.tasks = ['Class']
        train_dataset, valid_dataset, test_dataset = load_dataset_scaffold(args.data, args.dataset, args.seed,
                                                                           args.tasks)
        args.out_dim = 2 * len(args.tasks)
    elif args.dataset == 'bbbp':
        args.tasks = ['BBBP']
        train_dataset, valid_dataset, test_dataset = load_dataset_scaffold(args.data, args.dataset, args.seed,
                                                                           args.tasks)
        args.out_dim = 2 * len(args.tasks)
    else:  # Unknown dataset error
        raise Exception('Unknown dataset, please enter the correct --dataset option')

    args.in_dim = train_dataset.num_node_features
    args.edge_in_dim = train_dataset.num_edge_features
    weight = train_dataset.weights
    option = args.__dict__

    if not args.eval:
        model = Model(args.in_dim, args.edge_in_dim, hidden_dim=args.hid, depth=args.depth,
                      heads=args.heads, dropout=args.dropout, outdim=args.out_dim)
        trainer = Trainer(option, model, train_dataset, valid_dataset, test_dataset, weight=weight,
                          tasks_num=len(args.tasks))
        trainer.train()
        print('Testing...')
        trainer.load_best_ckpt()
        trainer.valid_iterations(mode='eval')
    else:
        ckpt = torch.load(args.load)
        option = ckpt['option']
        model = Model(option['in_dim'], option['edge_in_dim'], hidden_dim=option['hid'], depth=option['depth'],
                      heads=option['heads'], dropout=option['dropout'], outdim=option['out_dim'])
        if not os.path.exists(option['exp_path']): os.makedirs(option['exp_path'])
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        trainer = Trainer(option, model, train_dataset, valid_dataset, test_dataset, weight=weight,
                          tasks_num=len(args.tasks))
        trainer.valid_iterations(mode='eval')


if __name__ == '__main__':
    train()
