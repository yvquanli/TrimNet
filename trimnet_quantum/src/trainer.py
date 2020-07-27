import time
import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel

torch.backends.cudnn.enabled = True


class Trainer():
    def __init__(self, option, model,
                 train_dataset, valid_dataset, test_dataset=None):
        self.option = option   #; print(option)
        self.device = torch.device("cuda:{}".format(option['cuda_device']) \
                                       if torch.cuda.is_available() else "cpu")
        self.model = DataParallel(model, device_ids=self.option['parallel_devices']).to(self.device) \
            if option['parallel'] else model.to(self.device)

        # Setting the train valid and test data loader
        if self.option['parallel']:
            self.train_dataloader = DataListLoader(train_dataset, \
                                                   batch_size=self.option['train_batch'])
            self.valid_dataloader = DataListLoader(valid_dataset, batch_size=64)
            if test_dataset: self.test_dataloader = DataListLoader(test_dataset, batch_size=64)
        else:
            self.train_dataloader = DataLoader(train_dataset, \
                                               batch_size=self.option['train_batch'])
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=64)
            if test_dataset: self.test_dataloader = DataLoader(test_dataset, batch_size=64)

        # Setting the Adam optimizer with hyper-param
        self.criterion = torch.nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.option['lr'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7,
            patience=self.option['lr_scheduler_patience'], min_lr=1e-6
        )

        # other
        self.start = time.time()
        self.save_id = ''.join(random.sample('zyxwvutsrqponmlkjihgfedcba1234567890', 4))
        self.abs_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.ckpt_save_dir = os.path.join(self.abs_file_dir, 'ckpt', 'ckpts_{}_task{}_{}'.
                                          format(self.option['dataset'], self.option['task'],
                                                 self.save_id))
        self.log_save_path = os.path.join(self.abs_file_dir, 'log', 'log_{}_task{}_{}.txt'.
                                          format(self.option['dataset'], self.option['task'],
                                                 self.save_id))
        self.record_save_path = os.path.join(self.abs_file_dir, 'record', 'record_{}_task{}_{}.csv'.
                                             format(self.option['dataset'], self.option['task'],
                                                    self.save_id))

        os.system('mkdir -p log record {}'.format(self.ckpt_save_dir))
        self.records = {'trn_record': [], 'val_record': [], 'val_losses': [],
                        'best_ckpt': None}
        self.log(msgs=['\t{}:{}\n'.format(k, v) for k, v in self.option.items()])
        self.log('save id: {}'.format(self.save_id))
        self.log('train set num:{}    valid set num:{}    test set num: {}'.format(
            len(train_dataset), len(valid_dataset), len(test_dataset)))
        self.log("total parameters:" + str(sum([p.nelement() for p in self.model.parameters()])))
        self.log(msgs=str(model).split('\n'))

    def train_iterations(self):
        self.model.train()
        losses = []
        for i, data in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            if self.option['parallel']:
                sample_list = data  # data will be sample_list in parallel model
                output = self.model(sample_list)
                y = torch.cat([sample.y for sample in sample_list]).to(output.device)
                loss = self.criterion(output, y)
            else:
                data = data.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, data.y)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            if i % 100 == 0:
                self.log('\tbatch {} training loss: {:.5f}'.format(i, loss.item()), with_time=True)
        trn_loss = np.array(losses).mean()
        return trn_loss

    def valid_iterations(self, mode='valid'):
        self.model.eval()
        if mode == 'test': dataloader = self.test_dataloader
        if mode == 'valid': dataloader = self.valid_dataloader
        outputs = []
        ys = []
        with torch.no_grad():
            for data in dataloader:
                if self.option['parallel']:
                    sample_list = data  # data will be samplelist in parallel model
                    output = self.model(sample_list)
                    y = torch.cat([sample.y for sample in sample_list]).to(output.device)
                else:
                    data = data.to(self.device)
                    output = self.model(data)
                    y = data.y
                outputs.append(output.to('cpu'))
                ys.append(y.to('cpu'))
        val_loss = self.criterion(torch.cat(outputs), torch.cat(ys)).item()
        if mode == 'test': self.log('Test loss: {:.5f}'.format(val_loss))
        return val_loss

    def gen_test_batch(self):
        for batch in self.valid_dataloader:
            torch.save(batch, 'a_test_batch')
            break

    def train(self):
        self.log('Training start...')
        early_stop_cnt = 0
        for epoch in tqdm(range(self.option['train_epoch'])):
            trn_loss = self.train_iterations()
            val_loss = self.valid_iterations()
            self.scheduler.step(val_loss)
            lr_cur = self.scheduler.optimizer.param_groups[0]['lr']
            self.log('Epoch:{} trn_loss:{:.5f} val_loss:{:.5f} lr_cur:{:.7f}'.format(epoch, trn_loss, val_loss, lr_cur),
                     with_time=True)
            self.records['val_losses'].append(val_loss)
            self.records['val_record'].append([epoch, val_loss, lr_cur])
            self.records['trn_record'].append([epoch, trn_loss, lr_cur])
            if val_loss == np.array(self.records['val_losses']).min():
                self.save_model_and_records(epoch, trn_loss, val_loss)
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1
            if self.option['early_stop_patience'] > 0 and early_stop_cnt > self.option['early_stop_patience']:

                self.log('Early stop hitted!')
                break
        self.save_model_and_records(epoch, trn_loss, val_loss, final_save=True)

    def save_model_and_records(self, epoch, trn_loss, val_loss, final_save=False):
        if final_save:
            self.save_loss_records()
            file_name = 'Final_save{}_{}_{:.5f}_{:.5f}.ckpt'.format(
                self.option['task'], epoch, trn_loss, val_loss)
        else:
            file_name = 'task{}_{}_{:.5f}_{:.5f}.ckpt'.format(
                self.option['task'], epoch, trn_loss, val_loss)
            self.records['best_ckpt'] = file_name

        with open(os.path.join(self.ckpt_save_dir, file_name), 'wb') as f:
            torch.save({
                'option': self.option,
                'records': self.records,
                'model_state_dict': self.model.state_dict(),
            }, f)
        self.log('Model saved at epoch {}'.format(epoch))

    def save_loss_records(self):
        trn_record = pd.DataFrame(self.records['trn_record'],
                                  columns=['epoch', 'trn_loss', 'lr'])
        val_record = pd.DataFrame(self.records['val_record'],
                                  columns=['epoch', 'val_loss', 'lr'])
        ret = pd.DataFrame({
            'Epoch': trn_record['epoch'],
            'Traning MAE Loss': trn_record['trn_loss'],
            'Validation MAE Loss': val_record['val_loss'],
        })
        ret.to_csv(self.record_save_path)
        return ret

    def load_best_ckpt(self):
        ckpt_path = self.ckpt_save_dir + '/' + self.records['best_ckpt']
        self.log('The best ckpt is {}'.format(ckpt_path))
        self.load_ckpt(ckpt_path)

    def load_ckpt(self, ckpt_path):
        self.log('Ckpt loading: {}'.format(ckpt_path))
        ckpt = torch.load(ckpt_path)
        self.option = ckpt['option']
        self.records = ckpt['records']
        self.model.load_state_dict(ckpt['model_state_dict'])

    def log(self, msg=None, msgs=None, with_time=False):
        if with_time: msg = msg + ' time elapsed {:.2f} hrs ({:.1f} mins)'.format(
            (time.time() - self.start) / 3600.,
            (time.time() - self.start) / 60.
        )
        with open(self.log_save_path, 'a+') as f:
            if msgs:
                self.log('#' * 80)
                if '\n' not in msgs[0]: msgs = [m + '\n' for m in msgs]
                f.writelines(msgs);
                for x in msgs:
                    print(x, end='')
                self.log('#' * 80)
            if msg:  f.write(msg + '\n');  print(msg)


if __name__ == '__main__':
    trainer = Trainer(None, None, None, None)
