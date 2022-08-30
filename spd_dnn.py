#!/user/bin/env python
# -*- coding:utf-8 -*-
"""
@Time       : 2022/8/30 16:21
@Author     : zhushuli
@File       : spd_dnn.py
@DevTool    : PyCharm
"""
import torch
import torch.nn as nn
import numpy as np

__all__ = ['load_spd_dnn', 'EarlyStopping']


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size, use_relu=True, use_leaky=True):
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(in_size, out_size)
        self.use_relu = use_relu
        if use_relu:
            if use_leaky:
                self.relu = nn.LeakyReLU()
            else:
                self.relu = nn.ReLU()

    def forward(self, x):
        input = x.permute(0, 2, 1)  # b*f*s -> b*s*f
        out = self.fc(input).permute(0, 2, 1)
        if self.use_relu:
            return self.relu(out)
        return out


class BasicNetV2(nn.Module):
    def __init__(self, acc_sizes, gy_sizes, grv_sizes):
        super(BasicNetV2, self).__init__()
        # gravity embedding
        grv_layers = []
        for i in range(1, len(grv_sizes)):
            grv_layers.append(FCLayer(grv_sizes[i - 1], grv_sizes[i]))
        self.grv_ebd = nn.Sequential(*grv_layers)
        # acc embedding
        acc_layers = []
        for i in range(1, len(acc_sizes)):
            acc_layers.append(FCLayer(acc_sizes[i - 1], acc_sizes[i]))
        self.acc_ebd = nn.Sequential(*acc_layers)
        # gy embedding
        gy_layers = []
        for i in range(1, len(gy_sizes)):
            gy_layers.append(FCLayer(gy_sizes[i - 1], gy_sizes[i]))
        self.gy_ebd = nn.Sequential(*gy_layers)

    def forward(self, acc, gy):
        gvt_ebd = self.grv_ebd(torch.cat([acc, gy], dim=1))
        acc_ebd = self.acc_ebd(torch.cat([acc, gvt_ebd], dim=1))
        gy_ebd = self.gy_ebd(torch.cat([gy, gvt_ebd], dim=1))
        return torch.cat([acc_ebd, gy_ebd], dim=1)


class PdrSpdLSTMV2(nn.Module):
    def __init__(self, acc_sizes, gy_sizes, grv_sizes,
                 lstm_input_size, lstm_hidden_size, lstm_num_layers, fc_spd_sizes):
        super(PdrSpdLSTMV2, self).__init__()
        self.basic_net = BasicNetV2(acc_sizes, gy_sizes, grv_sizes)
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers)
        # speed fully connected
        spd_layers = []
        for i in range(1, len(fc_spd_sizes)):
            if i <= 2:
                spd_layers.append(FCLayer(fc_spd_sizes[i - 1], fc_spd_sizes[i], use_relu=True, use_leaky=False))
            else:
                spd_layers.append(FCLayer(fc_spd_sizes[i - 1], fc_spd_sizes[i], use_relu=False, use_leaky=False))
        self.fc_spd_out = nn.Sequential(*spd_layers)

    def forward(self, acc, gy, init_spd=None, src_mask=None):
        basic_res = self.basic_net(acc, gy)  # b*f*s
        lstm_res = self.lstm(basic_res.permute(2, 0, 1))[0].permute(1, 2, 0)  # s*b*f -> b*f*s
        if init_spd is not None:
            spd_out = self.fc_spd_out(torch.cat([lstm_res, init_spd], dim=1))
        else:
            spd_out = self.fc_spd_out(lstm_res)
        return spd_out


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args::wq
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: %d out of %d' % (self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.early_stop = False

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Validation loss decreased (%.6f --> %.6f).  Saving model ...' % (self.val_loss_min, val_loss))
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def load_spd_dnn(pretrained_model=None, trained_model=None, ftr_hz=50, collect_hz=250):
    mutiple = collect_hz // ftr_hz
    model = PdrSpdLSTMV2(acc_sizes=(12 * mutiple + 128, 64, 32), gy_sizes=(12 * mutiple + 128, 64, 32),
                         grv_sizes=(12 * mutiple + 12 * mutiple, 64, 128),
                         lstm_input_size=64, lstm_hidden_size=128, lstm_num_layers=1,
                         fc_spd_sizes=(128 + 1, 64, 32, 16, 1))

    if pretrained_model is not None and trained_model is None:
        print('Load pretrained model: %s' % pretrained_model)
        model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
    if trained_model is not None:
        print('Load trained model: %s' % trained_model)
        model.load_state_dict(torch.load(trained_model, map_location='cpu'))
    return model
