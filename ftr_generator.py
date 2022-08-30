#!/user/bin/env python
# -*- coding:utf-8 -*-
"""
@Time       : 2022/8/30 14:45
@Author     : zhushuli
@File       : ftr_generator.py
@DevTool    : PyCharm
"""
import numpy as np
from data_struct import *

__all__ = ['ftrGenerator']


class ftrConf():
    def __init__(self, conf_path):
        self.name_idx_map = {}
        self.idx_name_map = {}
        self.name_domain_map = {}
        self.name_dft_map = {}
        self.name_cache_map = {}

        with open(conf_path, 'r') as fd:
            for line in fd:
                _items = line.strip().split(' ')[0].split(',')
                ftr_name = _items[0]
                ftr_idx = int(_items[1])
                ftr_domain = (float(_items[2]), float(_items[3]))
                ftr_dft_val = float(_items[4])
                self.name_idx_map[ftr_name] = ftr_idx
                self.idx_name_map[ftr_idx] = ftr_name
                self.name_domain_map[ftr_name] = ftr_domain
                self.name_dft_map[ftr_name] = ftr_dft_val


class ftrGenerator():
    def __init__(self, conf_path='spd_dnn_feature_v1.0', ftr_hz=50, collect_hz=250):
        assert collect_hz % ftr_hz == 0
        self.ftr_hz = ftr_hz
        self.collect_hz = collect_hz
        self.__load_conf_file(conf_path)

    def __load_conf_file(self, conf_path):
        # self.ftr_conf = ftrConf(conf_path)
        self.func_map = {'mean': np.mean, 'std': np.std, 'min': np.min, 'max': np.max}

    def calc_imu_sec_ftr(self, sen_unit_list):
        assert len(sen_unit_list) == self.collect_hz

        sen_xs = [u.sen_x for u in sen_unit_list]
        sen_ys = [u.sen_y for u in sen_unit_list]
        sen_zs = [u.sen_z for u in sen_unit_list]

        ftr_list = []
        for data in [sen_xs, sen_ys, sen_zs]:
            for func_name, func in self.func_map.items():
                ftr_vals = [round(func(data[i:i + self.ftr_hz]), 4) for i in range(0, self.collect_hz, self.ftr_hz)]
                ftr_list.extend(ftr_vals)
        return ftr_list
