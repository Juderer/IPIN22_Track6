#!/user/bin/env python
# -*- coding:utf-8 -*-
"""
@Time       : 2022/8/30 17:36
@Author     : zhushuli
@File       : spd_dnn_evaluation.py
@DevTool    : PyCharm
"""
import os
import sys
import copy
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from spd_dnn import *
from data_struct import *
from ftr_generator import *
from coord_utils import *

def infer_bearing(sec_gyro_unit_list, crnt_brng, collect_hz=250):
    gy_ys = [u.sen_y for u in sec_gyro_unit_list]
    bearing_diff = np.sum(gy_ys) / collect_hz * -1.0
    return (crnt_brng + bearing_diff) % 360.0


def calibrate_sample_freq(imu_unit_list, collect_hz=250):
    if len(imu_unit_list) < collect_hz:
        imu_unit_list.extend([copy.deepcopy(imu_unit_list[-1])] * (collect_hz - len(imu_unit_list)))
    elif len(imu_unit_list) > collect_hz:
        imu_unit_list = imu_unit_list[-collect_hz:]
    else:
        pass
    return imu_unit_list


def infer_speed(sec_acc_unit_list, sec_gyro_unit_list, crnt_spd,
                model=None, ftr_generator=None, collect_hz=250):
    sec_acc_unit_list = calibrate_sample_freq(sec_acc_unit_list, collect_hz=collect_hz)
    sec_gyro_unit_lis = calibrate_sample_freq(sec_gyro_unit_list, collect_hz=collect_hz)
    acc_ftr_list = ftr_generator.calc_imu_sec_ftr(sec_acc_unit_list)
    gy_ftr_list = ftr_generator.calc_imu_sec_ftr(sec_gyro_unit_lis)

    acc = torch.tensor([[acc_ftr_list]], dtype=torch.float).permute(0, 2, 1)
    gy = torch.tensor([[gy_ftr_list]], dtype=torch.float).permute(0, 2, 1)
    init_spd = torch.tensor([[[crnt_spd]]], dtype=torch.float)
    pred = model(acc, gy, init_spd=init_spd)
    spd_diff = pred.cpu().detach().squeeze().numpy().tolist()

    return max(0.0, min(33.0, crnt_spd + spd_diff))


if __name__ == '__main__':
    ftr_hz = 50
    collect_hz = 250

    model = load_spd_dnn(trained_model='./spd_dnn_weight_ftrHz%d.pt' % ftr_hz)
    model.eval()

    ftr_generator = ftrGenerator(ftr_hz=ftr_hz, collect_hz=collect_hz)

    with open('./trials/IPIN2022_T7_TestingTrial02.txt', 'r') as fd:
        pre_gnss = None
        start_gnss = None
        gt_gnss_unit_list = []
        sec_acc_unit_list, sec_gyro_unit_list = [], []
        pred_arr = []
        for i, line in enumerate(fd):
            if pre_gnss is None and line.startswith('GNSS'):
                pre_gnss = gnssUnit(line)
                continue
            if start_gnss is None and line.startswith('GNSS'):
                start_gnss = gnssUnit(line, pre_gnss=pre_gnss)
                pre_gnss = start_gnss
            elif start_gnss is not None and line.startswith('GNSS'):
                gnss = gnssUnit(line, pre_gnss=pre_gnss)
                pre_gnss = gnss
                gt_gnss_unit_list.append(gnss)
                if len(pred_arr) > 0:
                    bearing = infer_bearing(sec_gyro_unit_list, pred_arr[-1][3])
                    speed = infer_speed(sec_acc_unit_list, sec_gyro_unit_list, pred_arr[-1][2],
                                        ftr_generator=ftr_generator, model=model)
                    lng, lat = geo_util.position_from_angle_and_dist(pred_arr[-1][0], pred_arr[-1][1],
                                                                     pred_arr[-1][3], pred_arr[-1][2])
                else:
                    bearing = infer_bearing(sec_gyro_unit_list, start_gnss.bearing)
                    speed = infer_speed(sec_acc_unit_list, sec_gyro_unit_list, start_gnss.loc_speed,
                                        ftr_generator=ftr_generator, model=model)
                    lng, lat = geo_util.position_from_angle_and_dist(start_gnss.lng_gcj02, start_gnss.lat_gcj02,
                                                                     start_gnss.bearing, start_gnss.loc_speed)
                pred_arr.append([lng, lat, speed, bearing])
                sec_acc_unit_list, sec_gyro_unit_list = [], []

            elif start_gnss is not None and line.startswith('ACCE'):
                sec_acc_unit_list.append(acceUnit(line))
            elif start_gnss is not None and line.startswith('GYRO'):
                sec_gyro_unit_list.append(gyroUnit(line))
            if i == 50000:
                break

    print(start_gnss)
    # print(gt_gnss_unit_list)
    # print(pred_arr)

    print([u.bearing for u in gt_gnss_unit_list])
    print([round(u[3], 2) for u in pred_arr])

    print([round(u.loc_speed, 2) for u in gt_gnss_unit_list])
    print([round(u[2], 2) for u in pred_arr])

    with open('evaluation_result.csv', 'w') as fd:
        fd.write('gtLng,gtLat,predLng,predLat\n')
        fd.write('%s,%s\n' % (start_gnss.loc_gcj02_str, start_gnss.loc_gcj02_str))
        for gt_gnss, pred_list in zip(gt_gnss_unit_list, pred_arr):
            fd.write('%s,%.6f,%.6f\n' % (gt_gnss.loc_gcj02_str, pred_list[0], pred_list[1]))
