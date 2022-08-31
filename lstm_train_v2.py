#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2022/4/13 17:45
@Author     : zhushuli
@DevTool    : PyCharm
@File       : lstm_train_v2.py
@CopyFrom   : lstm_train.py
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from RNN_Net import *
# from custom_loss_fn import *
from utils import *
from coord_utils import *


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

    @staticmethod
    def my_collate_fn(datas):
        pose_idx = []
        crnt_lon, crnt_lat, crnt_spd, crnt_brng = [], [], [], []
        acc, gy = [], []
        next_lon, next_lat, next_spd, next_brng = [], [], [], []
        for data in datas:
            pose_idx.append(data[0])
            crnt_lon.append(data[1])
            crnt_lat.append(data[2])
            crnt_spd.append(data[3])
            crnt_brng.append(data[4])
            next_lon.append(data[5])
            next_lat.append(data[6])
            next_spd.append(data[7])
            next_brng.append(data[8])
            acc.append(data[9])
            gy.append(data[10])
        pose_idx = torch.tensor(pose_idx).permute(0, 2, 1)
        crnt_lon = torch.tensor(crnt_lon).permute(0, 2, 1)
        crnt_lat = torch.tensor(crnt_lat).permute(0, 2, 1)
        crnt_spd = torch.tensor(crnt_spd).permute(0, 2, 1)
        crnt_brng = torch.tensor(crnt_brng).permute(0, 2, 1)
        next_lon = torch.tensor(next_lon).permute(0, 2, 1)
        next_lat = torch.tensor(next_lat).permute(0, 2, 1)
        next_spd = torch.tensor(next_spd).permute(0, 2, 1)
        next_brng = torch.tensor(next_brng).permute(0, 2, 1)
        acc = torch.tensor(acc).permute(0, 2, 1)
        gy = torch.tensor(gy).permute(0, 2, 1)
        return pose_idx, crnt_spd, crnt_brng, acc, gy, next_spd, next_brng


class MyDataset(Dataset):
    def __init__(self, input_file_paths, version=None):
        self.version = version
        self.sample_list = []
        for input_file_path in input_file_paths:
            with open(input_file_path, 'r') as fd:
                self.sample_list.extend([line.strip() for line in fd if len(line) > 1])

    def __getitem__(self, item):
        sample = self.sample_list[item]
        pose_idx_arr = []
        crnt_lon_arr, crnt_lat_arr, crnt_spd_arr, crnt_brng_arr = [], [], [], []
        acc_ftr_arr, gy_ftr_arr, grv_ftr_arr = [], [], []
        next_lon_arr, next_lat_arr, next_spd_arr, next_brng_arr = [], [], [], []
        for line in sample.strip().split('|'):
            _items = [float(x) for x in line.strip().split(',')]
            pose_idx_arr.append([_items[0]])
            crnt_lon_arr.append([_items[1]])
            crnt_lat_arr.append([_items[2]])
            crnt_spd_arr.append([_items[3]])
            crnt_brng_arr.append([_items[4]])
            next_lon_arr.append([_items[5]])
            next_lat_arr.append([_items[6]])
            next_spd_arr.append([_items[7]])
            next_brng_arr.append([_items[8]])
            if self.version is None:
                acc_ftr_arr.append(_items[9:759])
                gy_ftr_arr.append(_items[759:1509])
            elif self.version == 'v1.0':
                acc_ftr_arr.append(_items[9:21])
                gy_ftr_arr.append(_items[21:33])
            elif self.version == 'v2.0':
                acc_ftr_arr.append(_items[9:39])
                gy_ftr_arr.append(_items[39:69])
            else:
                raise ValueError('Feature version is illegal!')
        return pose_idx_arr, \
               crnt_lon_arr, crnt_lat_arr, crnt_spd_arr, crnt_brng_arr, \
               next_lon_arr, next_lat_arr, next_spd_arr, next_brng_arr, \
               acc_ftr_arr, gy_ftr_arr

    def __len__(self):
        return len(self.sample_list)


def calc_label(crnt_spd, crnt_brng, next_spd, next_brng, pred_type='vel_diff'):
    # dist_diff = (next_spd + crnt_spd) / 2
    # dist_diff = geo_util.haversine_formula_tensor(crnt_lon, crnt_lat, next_lon, next_lat)
    spd_diff = next_spd - crnt_spd
    brng_diff = torch.abs(next_brng - crnt_brng)
    brng_diff = brng_diff.min(360 - brng_diff)
    # 数据精度影响对正负的判断
    crnt_brng_np = crnt_brng.numpy()
    next_brng_np = next_brng.numpy()
    brng_diff_np = brng_diff.numpy()
    is_clockwise = np.round((crnt_brng_np + brng_diff_np) % 360, 2) == np.round(next_brng_np, 2)
    # 规定角度变化以顺时针方向为正、逆时针方向为负
    sign = torch.tensor(is_clockwise * 2 - 1)
    brng_diff = brng_diff * sign
    if pred_type == 'vel_val':
        return next_spd
    return spd_diff


def train_model(train_loader):
    train_loss_list = []
    for i, (pose_idx, crnt_spd, crnt_brng, acc, gy, next_spd, next_brng) in enumerate(train_loader):
        label = calc_label(crnt_spd, crnt_brng, next_spd, next_brng, pred_type='vel_diff')
        N, F, S = crnt_spd.shape
        init_spd = crnt_spd[:, :, 0:1].repeat(1, 1, S)
        # print('init speed ', init_spd)
        # print('acc ', acc)
        pred = model(acc, gy, init_spd=init_spd, pose=None)
        # loss = loss_fn(pred, label, pred_type='vel_diff')
        loss = loss_fn(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.item())
        # print('label ', label)
        # print('pred ', pred)
    return np.mean(train_loss_list)


def test_model(test_loader):
    test_loss_list = []
    for i, (pose_idx, crnt_spd, crnt_brng, acc, gy, next_spd, next_brng) in enumerate(test_loader):
        label = calc_label(crnt_spd, crnt_brng, next_spd, next_brng, pred_type='vel_diff')
        N, F, S = crnt_spd.shape
        init_spd = crnt_spd[:, :, 0:1].repeat(1, 1, S)
        # print('init speed ', init_spd)
        pred = model(acc, gy, init_spd=init_spd, pose=None)
        # loss = loss_fn(pred, label, pred_type='vel_diff')
        loss = loss_fn(pred, label)
        test_loss_list.append(loss.item())
    return np.mean(test_loss_list)

if __name__ == '__main__':
    phone_pose = sys.argv[1]  # flat, calling, pocket, None
    seq_len = int(sys.argv[2])
    ftr_version = None
    if len(sys.argv) > 3:
        ftr_version = sys.argv[3]
    print('phone_pose = %s, seq_len = %d, ftr_version = %s' % (phone_pose, seq_len, ftr_version))

    # 加载速度模型
    model = load_rnn_net(version=ftr_version)
    # model = load_rnn_net(version=ftr_version,
    #                      pretrained_model='./tmp_model/spdLSTM_weight_%s_seqLen%d_%s' % (phone_pose, seq_len, ftr_version))

    # 加载数据集
    # dataset = MyDataset(['./dataset/lstm_datasetv2_train_%s_seqLen%d_%s.txt' % (phone_pose, seq_len, ftr_version)],
    dataset = MyDataset(['./dataset/lstm_train_dataset_flat_4400.txt'],
                        version=ftr_version)
    train_loader = MultiEpochsDataLoader(batch_size=64,
                                         dataset=dataset,
                                         collate_fn=lambda x: MultiEpochsDataLoader.my_collate_fn(x),
                                         shuffle=True)
    # test_dataset = MyDataset(['./tmp_dataset/lstm_datasetv2_test_%s_seqLen%d_%s.txt' % (phone_pose, seq_len, ftr_version)],
    test_dataset = MyDataset(['./dataset/lstm_test_dataset_flat_500.txt'],
                             version=ftr_version)
    test_loader = MultiEpochsDataLoader(batch_size=64,
                                        dataset=test_dataset,
                                        collate_fn=lambda x: MultiEpochsDataLoader.my_collate_fn(x),
                                        shuffle=False)

    # 确定损失函数
    # loss_fn_obj = CustomLossFn(mult_factor=6)
    # loss_fn = loss_fn_obj.enlarge_only_vel_loss
    loss_fn = SmoothL1Loss()
    # 确定优化函数
    optimizer = Adam(model.parameters(), lr=1.8e-3)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.50)
    early_stopping = EarlyStopping(patience=7,
                                   verbose=True,
                                   # path='./model/spdLSTM_weight_%s_seqLen%d_%s3' % (phone_pose, seq_len, ftr_version))
                                   path='./model/model20.pt')

    EPOCHS = 80
    for epoch in range(1, EPOCHS + 1):
        print('=' * 17 + 'epoch: {epoch}'.format(epoch=str(epoch)) + '=' * 17)
        model.train()
        train_loss = train_model(train_loader)
        print('{epoch_idx} epoch train loss: {epoch_loss:.4f}' \
              .format(epoch_idx=epoch, epoch_loss=train_loss))
        scheduler.step()
        model.eval()
        test_loss = test_model(test_loader)
        print('{epoch_idx} epoch test loss: {epoch_loss:.4f}' \
              .format(epoch_idx=epoch, epoch_loss=test_loss))
        early_stopping(test_loss, model)
        if early_stopping.early_stop and epoch * 2 > EPOCHS:
        # if early_stopping.early_stop:
            break
    # torch.save(model.state_dict(), './model/model1.pt')
