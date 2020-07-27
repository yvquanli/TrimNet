import os
import sys
import warnings
from trainer import Trainer
from utils import seed_torch
from dataset import load_dataset
from model import TrimNet
from utils import seed_torch

warnings.filterwarnings("ignore")

task = 0 if len(sys.argv) < 2 else int(sys.argv[1])  # [0~12]
depth = 3 if len(sys.argv) < 3 else int(sys.argv[2])
cuda_device = 0 if len(sys.argv) < 4 else int(sys.argv[3])
seed = 1234 if len(sys.argv) < 5 else int(sys.argv[4])

# seed_torch(seed)

option = {
    'dataset': 'QM9',
    'task': task,  # [0~11]
    'data_path': '../dataset/',
    'code_file_path': os.path.abspath(__file__),

    'train_epoch': 200,
    'train_batch': 64,
    'lr': 1e-3,
    'lr_scheduler_patience': 5,
    'early_stop_patience': -1,  # -1 for no early stop
    'depth': depth,

    'cuda_device': cuda_device,  # 1
    'seed': seed,
    'parallel': False,
    'parallel_devices': [0, 1, 2, 3],  # works when parallel=True
}

print('Loading dataset...')
train_dataset, valid_dataset, test_dataset = load_dataset(option['data_path'], option['task'])
in_dim = train_dataset.num_node_features
edge_in_dim = train_dataset.num_edge_features

print('Training init...')
model = TrimNet(in_dim, edge_in_dim, depth=option['depth'])
trainer = Trainer(option, model, train_dataset, valid_dataset, test_dataset)
# trainer.gen_test_batch()
trainer.train()

print('Testing...')
trainer.load_best_ckpt()
trainer.valid_iterations(mode='test')
