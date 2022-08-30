#!/user/bin/env python
# -*- coding:utf-8 -*-
"""
@Time       : 2022/8/30 09:46
@Author     : zhushuli
@File       : spd_dnn_dataset_produce.py
@DevTool    : PyCharm
"""
import os
import sys
import copy
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_struct import *
from ftr_generator import *


def load_testing_data(path):
    unit_list = []
    with open(path, 'r') as fd:
        print('read %s' % path)
        pre_gnss = None
        for i, line in enumerate(fd):
            if line.startswith('GNSS'):
                gnss = gnssUnit(line, pre_gnss=pre_gnss)
                unit_list.append(gnss)
                pre_gnss = gnss
            elif line.startswith('ACCE'):
                unit_list.append(acceUnit(line))
            elif line.startswith('GYRO'):
                unit_list.append(gyroUnit(line))
            else:
                continue
    print('total count: %d' % len(unit_list))
    return sorted(unit_list, key=lambda x: x.app_ts, reverse=False)


def align_sec_data(unit_list):
    sec_unit_arr = []
    sec_unit_list = []
    flag = False
    for unit in unit_list:
        if isinstance(unit, gnssUnit) and len(sec_unit_list) == 0 and not flag:
            sec_unit_list.append(unit)
            flag = True
        elif flag and isinstance(unit, (acceUnit, gyroUnit)):
            sec_unit_list.append(unit)
        elif isinstance(unit, gnssUnit) and len(sec_unit_list) > 0 and flag:
            sec_unit_list.append(unit)
            if sec_unit_list[-1].app_ts - sec_unit_list[0].app_ts < 1.2 and \
                    geo_util.distance(sec_unit_list[0].lng, sec_unit_list[0].lat,
                                      sec_unit_list[-1].lng, sec_unit_list[-1].lat) < 33 and \
                    0 < sec_unit_list[0].loc_speed < 33 and 0 < sec_unit_list[-1].loc_speed < 33 and \
                    sec_unit_list[0].accuracy <= 15 and sec_unit_list[-1].accuracy <= 15:
                sec_unit_arr.append(sec_unit_list)
            sec_unit_list = [unit]
        else:
            continue
    print('align %d valid second data' % len(sec_unit_arr))
    return sec_unit_arr


def calibrate_sample_freq(imu_unit_list, collect_hz=250):
    if len(imu_unit_list) < collect_hz:
        imu_unit_list.extend([copy.deepcopy(imu_unit_list[-1])] * (collect_hz - len(imu_unit_list)))
    elif len(imu_unit_list) > collect_hz:
        imu_unit_list = imu_unit_list[-collect_hz:]
    else:
        pass
    return imu_unit_list


def gen_dnn_training_dataset(sec_unit_arr, ftr_generator=None):
    sample_str_list = []
    for sec_unit_list in sec_unit_arr:
        crnt_gnss_unit, next_gnss_unit = sec_unit_list[0], sec_unit_list[-1]
        acce_unit_list, gyro_unit_list = [], []
        for i in range(1, len(sec_unit_list)):
            if isinstance(sec_unit_list[i], acceUnit):
                acce_unit_list.append(sec_unit_list[i])
            elif isinstance(sec_unit_list[i], gyroUnit):
                gyro_unit_list.append(sec_unit_list[i])
            else:
                continue
        acce_unit_list = calibrate_sample_freq(acce_unit_list)
        gyro_unit_list = calibrate_sample_freq(gyro_unit_list)

        acce_ftr_list = ftr_generator.calc_imu_sec_ftr(acce_unit_list)
        gyro_ftr_list = ftr_generator.calc_imu_sec_ftr(gyro_unit_list)
        acce_ftr_str = ','.join(['%.4f' % x for x in acce_ftr_list])
        gyro_ftr_str = ','.join(['%.4f' % x for x in gyro_ftr_list])

        sample_str = '%.6f,%.6f,%.2f,%.2f;%.6f,%.6f,%.2f,%.2f;%s;%s' \
                     % (crnt_gnss_unit.lng, crnt_gnss_unit.lat, crnt_gnss_unit.loc_speed, crnt_gnss_unit.bearing,
                        next_gnss_unit.lng, next_gnss_unit.lat, next_gnss_unit.loc_speed, crnt_gnss_unit.bearing,
                        acce_ftr_str, gyro_ftr_str)
        sample_str_list.append(sample_str)
    print('sample_str_cnt = %d' % len(sample_str_list))
    return sample_str_list


if __name__ == '__main__':
    ftr_hz = 50
    collect_hz = 250
    if len(sys.argv) > 1:
        ftr_hz = int(sys.argv[1])

    assert collect_hz % ftr_hz == 0
    print('ftr_hz = %d, collect_hz = %d' % (ftr_hz, collect_hz))

    ftr_generator = ftrGenerator(ftr_hz=ftr_hz, collect_hz=collect_hz)

    testing_path_list = ['./trials/IPIN2022_T7_TestingTrial01.txt',
                         './trials/IPIN2022_T7_TestingTrial02.txt']
    sample_str_list = []
    for path in testing_path_list:
        unit_list = load_testing_data(path)
        sec_unit_arr = align_sec_data(unit_list)
        sample_str_list.extend(gen_dnn_training_dataset(sec_unit_arr, ftr_generator=ftr_generator))
        # break

    training_fd = open('./spd_dnn_training_dataset_ftrHz%d.txt' % ftr_hz, 'w')
    testing_fd = open('./spd_dnn_testing_dataset_ftrHz%d.txt' % ftr_hz, 'w')
    random.shuffle(sample_str_list)
    for i, sample in enumerate(sample_str_list, 1):
        if i % 4 == 0:
            testing_fd.write(sample + '\n')
        else:
            training_fd.write(sample + '\n')
    training_fd.close()
    testing_fd.close()
