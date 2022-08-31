#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2020/08/16 11:50
@Author     : Zhu Shuli
@File       : Utils.py
@DevTool    : PyCharm
@CopyFrom   : https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
"""
import os
import torch
import math
import numpy as np
from coord_utils import geo_util


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


class ComTools(object):
    def __init__(self):
        pass

    @staticmethod
    def calc_bearing_change(last_bearing, crnt_bearing):
        bearing_diff = abs(crnt_bearing - last_bearing)
        bearing_diff = min(bearing_diff, 360 - bearing_diff)
        if (last_bearing + bearing_diff) % 360 == crnt_bearing:
            return round(bearing_diff, 4)  # 顺时针为正
        else:
            return round(bearing_diff, 4) * -1  # 逆时针为负

    @staticmethod
    def rewrite_spd(crnt_lon, crnt_lat, crnt_spd, next_lon, next_lat, next_spd, phone_pose=None, adjusted=True):
        if phone_pose and phone_pose == 'flat':
            return crnt_spd, next_spd
        if not adjusted:
            return crnt_spd, next_spd
        sec_dist = geo_util.haversine_formula(crnt_lon, crnt_lat, next_lon, next_lat)
        if sec_dist * 2 <= crnt_spd + next_spd:
            return crnt_spd, next_spd
        comp = (sec_dist * 2 - crnt_spd - next_spd) / 2.0
        comp = min(comp, 2.0 - max(crnt_spd, next_spd))
        return min(crnt_spd + comp, 2.0), min(next_spd + comp, 2.0)

    @staticmethod
    def sigmoid_func(x, alpha=1):
        val = (math.e ** (x / alpha) - 1) / (math.e ** (x / alpha))
        return val

    @staticmethod
    def traverse_dir(path, file_list=None, file_suffix='txt'):
        # print("tra ", path)
        if file_list is None or not isinstance(file_list, list):
            file_list = []
        if os.path.isfile(path):
            if path.endswith(file_suffix):
                file_list.append(path)
        elif os.path.isdir(path):
            for filename in os.listdir(path):
                ComTools.traverse_dir('%s/%s' % (path, filename), file_list=file_list, file_suffix=file_suffix)
            return sorted(file_list)


if __name__ == '__main__':
    # 测试用例
    res = ComTools.traverse_dir('./dataset/220407_Wground_recovery')
    print(res)
