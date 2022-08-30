#!/user/bin/env python
# -*- coding:utf-8 -*-
"""
@Time       : 2022/8/30 16:20
@Author     : zhushuli
@File       : spd_dnn_train.py
@DevTool    : PyCharm
"""
import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import SmoothL1Loss
from torch.optim.lr_scheduler import StepLR

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from spd_dnn import *


class _RepeatSampler(object):
    """ 一直repeat的sampler """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class MultiEpochsDataLoader(DataLoader):
    """ 多epoch训练时，DataLoader对象不用重新建立线程和batch_sampler对象，以节约每个epoch的初始化时间 """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class MyDataset(Dataset):
    def __init__(self, input_file_paths):
        self.sample_list = []
        for input_file_path in input_file_paths:
            with open(input_file_path, 'r') as fd:
                self.sample_list.extend([line.strip() for line in fd if len(line) > 1])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, item):
        sample = self.sample_list[item]
        crnt_spd_arr, next_spd_arr = [], []
        acc_ftr_arr, gy_ftr_arr = [], []
        for line in sample.split('|'):
            crnt_gnss_str, next_gnss_str, acc_ftr_str, gy_ftr_str = line.split(';')
            crnt_lng, crnt_lat, crnt_spd, crnt_brng = [float(x) for x in crnt_gnss_str.split(',')]
            next_lng, next_lat, next_spd, next_brng = [float(x) for x in next_gnss_str.split(',')]
            acc_ftr_list = [float(x) for x in acc_ftr_str.split(',')]
            gy_ftr_list = [float(x) for x in gy_ftr_str.split(',')]

            crnt_spd_arr.append([crnt_spd])
            next_spd_arr.append([next_spd])
            acc_ftr_arr.append(acc_ftr_list)
            gy_ftr_arr.append(gy_ftr_list)
        return crnt_spd_arr, next_spd_arr, acc_ftr_arr, gy_ftr_arr

    @staticmethod
    def my_collate_fn(datas):
        crnt_spd, next_spd = [], []
        acc, gy = [], []
        for data in datas:
            crnt_spd.append(data[0])
            next_spd.append(data[1])
            acc.append(data[2])
            gy.append(data[3])
        crnt_spd = torch.tensor(crnt_spd).permute(0, 2, 1)
        next_spd = torch.tensor(next_spd).permute(0, 2, 1)
        acc = torch.tensor(acc).permute(0, 2, 1)
        gy = torch.tensor(gy).permute(0, 2, 1)
        return crnt_spd, next_spd, acc, gy


def calc_label(crnt_spd, next_spd, pred_type='vel_diff'):
    spd_diff = next_spd - crnt_spd
    if pred_type == 'vel_val':
        return next_spd
    return spd_diff


def train_model(training_loader):
    train_loss_list = []
    for i, (crnt_spd, next_spd, acc, gy) in enumerate(training_loader):
        label = calc_label(crnt_spd, next_spd, pred_type='vel_diff')
        N, F, S = crnt_spd.shape
        init_spd = crnt_spd[:, :, 0:1].repeat(1, 1, S)
        pred = model(acc, gy, init_spd=init_spd)
        # loss = loss_fn(pred, label, pred_type='vel_diff')
        loss = loss_fn(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.item())
    return np.mean(train_loss_list)


def test_model(testing_loader):
    train_loss_list = []
    for i, (crnt_spd, next_spd, acc, gy) in enumerate(testing_loader):
        label = calc_label(crnt_spd, next_spd, pred_type='vel_diff')
        N, F, S = crnt_spd.shape
        init_spd = crnt_spd[:, :, 0:1].repeat(1, 1, S)
        pred = model(acc, gy, init_spd=init_spd)
        loss = loss_fn(pred, label)
        train_loss_list.append(loss.item())
    return np.mean(train_loss_list)


if __name__ == '__main__':
    ftr_hz = 50
    collect_hz = 250
    if len(sys.argv) > 1:
        ftr_hz = int(sys.argv[1])

    assert collect_hz % ftr_hz == 0
    print('ftr_hz = %d, collect_hz = %d' % (ftr_hz, collect_hz))

    training_dataset = MyDataset(['./spd_dnn_training_dataset_ftrHz%d.txt' % ftr_hz])
    training_loader = MultiEpochsDataLoader(batch_size=64,
                                            dataset=training_dataset,
                                            collate_fn=MyDataset.my_collate_fn,
                                            shuffle=True)
    testing_dataset = MyDataset(['./spd_dnn_testing_dataset_ftrHz%d.txt' % ftr_hz])
    testing_loader = MultiEpochsDataLoader(batch_size=64,
                                           dataset=testing_dataset,
                                           collate_fn=MyDataset.my_collate_fn,
                                           shuffle=False)
    model = load_spd_dnn(ftr_hz=ftr_hz, collect_hz=collect_hz)

    loss_fn = SmoothL1Loss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.95)
    early_stopping = EarlyStopping(patience=7, verbose=True, path='./spd_dnn_weight_ftrHz%d.pt' % ftr_hz)

    EPOCHS = 100
    for epoch in range(1, EPOCHS + 1):
        print('=' * 17 + 'epoch: {epoch}'.format(epoch=str(epoch)) + '=' * 17)
        model.train()
        train_loss = train_model(training_loader)
        print('{epoch_idx} epoch train loss: {epoch_loss:.4f}' \
              .format(epoch_idx=epoch, epoch_loss=train_loss))
        scheduler.step()
        model.eval()
        test_loss = test_model(testing_loader)
        print('{epoch_idx} epoch test loss: {epoch_loss:.4f}' \
              .format(epoch_idx=epoch, epoch_loss=test_loss))
        early_stopping(test_loss, model)
        if early_stopping.early_stop and epoch * 2 > EPOCHS:
            break
