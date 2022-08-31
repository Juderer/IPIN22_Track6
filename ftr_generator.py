#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2021/08/23 16:00
@Author     : Zhu Shuli
@File       : ftr_generator.py
@DevTool    : PyCharm
@Desc       : 由传感器原始数据计算特征
"""
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fft_extract import *


class RnnFtrGenerator(object):
    def __init__(self, phone_pose='flat', ftr_ver_list=['v1.0'], ftr_hz=50, collect_hz=50):
        """
        Args
            phone_pose: 手机姿态
            ftr_ver_list: 特征版本号列表
            ftr_hz: 按赫兹计算秒内特征
        """
        self.phone_pose = phone_pose
        self.ftr_hz = ftr_hz
        self.collect_hz = collect_hz
        assert self.collect_hz % self.ftr_hz == 0
        self.__load_conf_file(phone_pose=phone_pose, ftr_ver_list=ftr_ver_list, ftr_hz=ftr_hz)

    def __load_conf_file(self, phone_pose='flat', ftr_ver_list=None, ftr_hz=50):
        """ 加载特征配置文件 """
        self.ftr_version_map = {}
        for version in ftr_ver_list:
            self.ftr_version_map[version] = {'name_idx_map': {}, 'idx_name_map': {},
                                             'name_domain_map': {}, 'name_dft_map': {},
                                             'name_cache_map': {}}
            with open('rnn_%s_feature_%s_ftrHz%d' % (phone_pose, version, ftr_hz), 'r') as fd:
                for line in fd:
                    _items = line.strip().split(' ')[0].split(',')
                    ftr_name = _items[0]
                    ftr_idx = int(_items[1])
                    ftr_domain = (float(_items[2]), float(_items[3]))
                    ftr_dft_val = float(_items[4])
                    self.ftr_version_map[version]['name_idx_map'][ftr_name] = ftr_idx
                    self.ftr_version_map[version]['idx_name_map'][ftr_idx] = ftr_name
                    self.ftr_version_map[version]['name_domain_map'][ftr_name] = ftr_domain
                    self.ftr_version_map[version]['name_dft_map'][ftr_name] = ftr_dft_val
                    self.ftr_version_map[version]['name_cache_map'][ftr_name] = []

    def __set_ftr_dft_val(self, ftr_list, version='v1.0'):
        for name, val in self.ftr_version_map[version]['name_dft_map'].items():
            ftr_list[self.ftr_version_map[version]['name_idx_map'][name]] = val

    def calc_sec_ftr_by_sen_data(self, sen_unit_list, version='v1.0'):
        ftr_list = [0.0] * len(self.ftr_version_map[version]['name_idx_map'])
        self.__set_ftr_dft_val(ftr_list, version=version)

        imu_data_map = {'acc': {'x': [u.accx for u in sen_unit_list],
                                'y': [u.accy for u in sen_unit_list],
                                'z': [u.accz for u in sen_unit_list]},
                        'gy': {'x': [u.gyrx for u in sen_unit_list],
                               'y': [u.gyry for u in sen_unit_list],
                               'z': [u.gyrz for u in sen_unit_list]},
                        'grv': {'x': [u.gamervx for u in sen_unit_list],
                                'y': [u.gamervy for u in sen_unit_list],
                                'z': [u.gamervz for u in sen_unit_list],
                                's': [u.gamervs for u in sen_unit_list]}}
        sen_type_list = ['acc', 'gy', 'grv']
        axis_list = ['x', 'y', 'z', 's']
        func_map = {'mean': np.mean, 'std': np.std, 'min': np.min, 'max': np.max,
                    'fft_mean': fft_ftr_gen.fft_mean, 'fft_std': fft_ftr_gen.fft_std,
                    'fft_entropy': fft_ftr_gen.fft_entropy, 'fft_energy': fft_ftr_gen.fft_energy,
                    'fft_skew': fft_ftr_gen.fft_skew, 'fft_kurt': fft_ftr_gen.fft_kurt}
        for sen_type in sen_type_list:
            for axis in axis_list:
                if axis not in imu_data_map[sen_type]:
                    continue
                data = imu_data_map[sen_type][axis]
                for func_name, func in func_map.items():
                    ftr_name = '%s_%s_%s' % (sen_type, axis, func_name)
                    if ftr_name not in self.ftr_version_map[version]['name_idx_map'].keys():
                        continue
                    # ftr_vals = [round(func(data[i:i + ftr_hz]), 4) for i in range(0, self.collect_hz, self.ftr_hz)]
                    ftr_vals = []
                    for i in range(0, self.collect_hz, self.ftr_hz):
                        ftr_val = min(max(func(data[i:i + self.ftr_hz]),
                                          self.ftr_version_map[version]['name_domain_map'][ftr_name][0]),
                                      self.ftr_version_map[version]['name_domain_map'][ftr_name][1])
                        ftr_vals.append(round(ftr_val, 4))
                    ftr_list[self.ftr_version_map[version]['name_idx_map'][ftr_name]] = ftr_vals
                    self.ftr_version_map[version]['name_cache_map'][ftr_name].extend(ftr_vals)

                    # ftr_val = min(max(func(data),
                    #                   self.ftr_version_map[version]['name_domain_map'][ftr_name][0]),
                    #               self.ftr_version_map[version]['name_domain_map'][ftr_name][1])
                    # ftr_list[self.ftr_version_map[version]['name_idx_map'][ftr_name]] = round(ftr_val, 4)
        return ftr_list

    def __calc_feature_domain__(self, ftr_val_list):
        median_val = np.median(ftr_val_list)
        mean_val = np.mean(ftr_val_list)
        perc1_val = np.percentile(ftr_val_list, 1)
        perc5_val = np.percentile(ftr_val_list, 5)
        perc95_val = np.percentile(ftr_val_list, 95)
        perc99_val = np.percentile(ftr_val_list, 99)
        return perc1_val, perc5_val, median_val, mean_val, perc95_val, perc99_val

    def calc_ftr_domain(self, version='v1.0'):
        idx_name_pair_list = sorted(self.ftr_version_map[version]['idx_name_map'].items(), key=lambda x: x[0])
        ftr_domain_conf_fd = open('rnn_%s_feature_%s_ftrHz%d' % (self.phone_pose, version, self.ftr_hz), 'w')
        for idx, name in idx_name_pair_list:
            ftr_val_list = self.ftr_version_map[version]['name_cache_map'][name]
            _, min_val, dft_val, _, max_val, _ = self.__calc_feature_domain__(ftr_val_list)
            ftr_domain_conf_fd.write('%s,%d,%.2f,%.2f,%.2f\n' % (name, idx, min_val, max_val, dft_val))
        ftr_domain_conf_fd.close()

    def plot_ori_and_ftr_data(self, sen_unit_arr, version='v1.0', imu='acc', axis='x'):
        """ 可视化原始数据与特征
        Args
            sen_unit_list: SenUnit数组(传感器原始数据)
            version: 特征版本号
            imu: 加速度计与陀螺仪之一['acc', 'gy']
            axis: IMU某一轴['x', 'y', 'z']
        """
        ftr_arr = []
        picked_imu_arr = []
        for sen_unit_list in sen_unit_arr:
            ftr_arr.append(self.calc_sec_ftr_by_sen_data(sen_unit_list, version=version))
            picked_imu_arr.append([u.get_by_imu_axis(imu=imu, axis=axis) for u in sen_unit_list])

        hz_imu_list = []
        hz_ftr_mean_list, hz_ftr_std_list, hz_ftr_min_list, hz_ftr_max_list = [], [], [], []
        for picked_imu_list, ftr_list in zip(picked_imu_arr, ftr_arr):
            hz_imu_list.extend(picked_imu_list)
            for i, ftr in enumerate(ftr_list):
                if '%s_%s_mean' % (imu, axis) in self.ftr_version_map[version]['idx_name_map'][i]:
                    hz_ftr_mean_list.extend([ftr] * len(picked_imu_list))
                elif '%s_%s_std' % (imu, axis) in self.ftr_version_map[version]['idx_name_map'][i]:
                    hz_ftr_std_list.extend([ftr] * len(picked_imu_list))
                elif '%s_%s_min' % (imu, axis) in self.ftr_version_map[version]['idx_name_map'][i]:
                    hz_ftr_min_list.extend([ftr] * len(picked_imu_list))
                elif '%s_%s_max' % (imu, axis) in self.ftr_version_map[version]['idx_name_map'][i]:
                    hz_ftr_max_list.extend([ftr] * len(picked_imu_list))
                else:
                    pass

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(hz_imu_list, label='ori_%s_%s' % (imu, axis), linewidth=1.5, color='IndianRed',)
        ax.plot(hz_ftr_mean_list, label='ftr_mean', linewidth=2.0, color='Teal')
        ax.plot(hz_ftr_std_list, label='ftr_std', linewidth=2.0, color='DarkBlue')
        ax.plot(hz_ftr_min_list, label='ftr_min', linewidth=2.0, color='YellowGreen')
        ax.plot(hz_ftr_max_list, label='ftr_max', linewidth=2.0, color='Purple')

        ax.set_xlabel('Time ($Hz$)', fontsize=24)
        ax.set_ylabel('Value', fontsize=24)
        ax.set_title('Time Domain Features', fontsize=30)

        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('./images/rnn_ftr%s_%s_%s.png' % (version, imu, axis), dpi=400)
        plt.show()


# class XgbFtrGenerator(object):
#     def __init__(self, conf_file):
#         self.conf_file = conf_file
#         if conf_file is not None or len(conf_file) != 0:
#             self.__load_conf_file(conf_file)
#
#     def __load_conf_file(self, conf_file):
#         # 读取配置文件
#         self.ftr_name_idx_map = {}
#         self.ftr_idx_name_map = {}
#         self.ftr_name_domain_map = {}
#         self.ftr_name_dft_map = {}
#         with open(conf_file, 'r') as fd:
#             for line in fd:
#                 _items = line.strip().split(' ')[0].split(',')
#                 ftr_name = _items[0]
#                 ftr_idx = int(_items[1])
#                 ftr_domain = (float(_items[2]), float(_items[3]))
#                 ftr_dft_val = float(_items[4])
#                 self.ftr_name_idx_map[ftr_name] = ftr_idx
#                 self.ftr_idx_name_map[ftr_idx] = ftr_name
#                 self.ftr_name_domain_map[ftr_name] = ftr_domain
#                 self.ftr_name_dft_map[ftr_name] = ftr_dft_val
#
#     def __set_ftr_as_dft_val(self, ftr_list):
#         for name, val in self.ftr_name_dft_map.items():
#             ftr_list[self.ftr_name_idx_map[name]] = val
#
#     def calc_sec_ftr_by_sensor(self, sen_unit_list):
#         ftr_list = [0.0] * len(self.ftr_name_idx_map)
#         self.__set_ftr_as_dft_val(ftr_list)
#
#         imu_data_map = {'acc': {'x': [u.accx for u in sen_unit_list],
#                                 'y': [u.accy for u in sen_unit_list],
#                                 'z': [u.accz for u in sen_unit_list]},
#                         'gy': {'x': [u.gyrx for u in sen_unit_list],
#                                'y': [u.gyry for u in sen_unit_list],
#                                'z': [u.gyrz for u in sen_unit_list]}}
#         sen_type_list = ['acc', 'gy']
#         axis_list = ['x', 'y', 'z']
#         func_map = {'mean': np.mean, 'min': np.min, 'max': np.max, 'var': np.var}
#         for sen_type in sen_type_list:
#             for axis in axis_list:
#                 data = imu_data_map[sen_type][axis]
#                 for func_name, func in func_map.items():
#                     ftr_name = '%s_%s_%s' % (sen_type, func_name, axis)
#                     if ftr_name not in self.ftr_name_idx_map.keys():
#                         continue
#                     ftr_val = min(max(func(data), self.ftr_name_domain_map[ftr_name][0]),
#                                   self.ftr_name_domain_map[ftr_name][1])
#                     ftr_list[self.ftr_name_idx_map[ftr_name]] = round(ftr_val, 4)
#
#         ftr_name = 'itv_len'
#         if ftr_name in self.ftr_name_idx_map.keys():
#             hz_cnt = len(sen_unit_list)
#             ftr_val = min(max(hz_cnt / 50, self.ftr_name_domain_map[ftr_name][0]),
#                           self.ftr_name_domain_map[ftr_name][1])
#             ftr_list[self.ftr_name_idx_map[ftr_name]] = round(ftr_val, 4)
#         return ftr_list
#
#     def calc_sec_ftr_by_imu_data(self, acc_list=None, gyr_list=None):
#         ftr_list = [0.0] * len(self.ftr_name_idx_map)
#         self.__set_ftr_as_dft_val(ftr_list)
#
#         if acc_list is None and gyr_list is None:
#             return ftr_list
#
#         imu_data_map = {}
#         sen_type_list = []
#         if acc_list is not None:
#             accx_list, accy_list, accz_list = acc_list[0::3], acc_list[1::3], acc_list[2::3]
#             imu_data_map['acc'] = {'x': accx_list, 'y': accy_list, 'z': accz_list}
#             sen_type_list.append('acc')
#         if gyr_list is not None:
#             gyrx_list, gyry_list, gyrz_list = gyr_list[0::3], gyr_list[1::3], gyr_list[2::3]
#             imu_data_map['gy'] = {'x': gyrx_list, 'y': gyry_list, 'z': gyrz_list}
#             sen_type_list.append('gy')
#
#         axis_list = ['x', 'y', 'z']
#         func_map = {'mean': np.mean, 'min': np.min, 'max': np.max, 'var': np.var}
#         for sen_type in sen_type_list:
#             for axis in axis_list:
#                 data = imu_data_map[sen_type][axis]
#                 for func_name, func in func_map.items():
#                     ftr_name = '%s_%s_%s' % (sen_type, func_name, axis)
#                     if ftr_name not in self.ftr_name_idx_map.keys():
#                         continue
#                     ftr_val = min(max(func(data), self.ftr_name_domain_map[ftr_name][0]),
#                                   self.ftr_name_domain_map[ftr_name][1])
#                     ftr_list[self.ftr_name_idx_map[ftr_name]] = round(ftr_val, 4)
#
#         ftr_name = 'itv_len'
#         if ftr_name in self.ftr_name_idx_map.keys():
#             hz_cnt = len(acc_list) // 3
#             ftr_val = min(max(hz_cnt / 50, self.ftr_name_domain_map[ftr_name][0]),
#                           self.ftr_name_domain_map[ftr_name][1])
#             ftr_list[self.ftr_name_idx_map[ftr_name]] = round(ftr_val, 4)
#         return ftr_list


if __name__ == '__main__':
    rnn_ftr_gen = RnnFtrGenerator()

    # 读取数据
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from data_struct import *
    file_path = '/Users/zhushuli/Desktop/PDR_dataset/220407_Wground_recovery/phone1/user8/2022-04-08 15-47-51.csv'
    ts_sen_map = {}
    fd = open(file_path, 'r')
    for i, line in enumerate(fd):
        if i == 0:
            continue
        sen_unit = SenUnitV2(line)
        if sen_unit.ts not in ts_sen_map:
            ts_sen_map[sen_unit.ts] = [sen_unit]
        else:
            ts_sen_map[sen_unit.ts].append(sen_unit)
    fd.close()

    sen_unit_arr = [u[1] for u in sorted(ts_sen_map.items(), key=lambda x: x[0])]
    rnn_ftr_gen.plot_ori_and_ftr_data(sen_unit_arr[:10], version='v1.0', imu='acc', axis='x')
