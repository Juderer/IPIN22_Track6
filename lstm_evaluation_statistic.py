#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2022/4/13 19:49
@Author     : zhushuli
@DevTool    : PyCharm
@File       : lstm_evaluation_statistic.py
@CopyFrom   : lstm_eval_ftr_produce.py, lstm_evaluation.py
"""
import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from data_struct import *
from coord_utils import *
from utils import *
from RNN_Net import *
from ftr_generator import *


def list_eval_files(dft_path='./dataset/220407_Wground_recovery', pose='flat'):
    # print('path', dft_path)
    dataset_filepath_list = ComTools.traverse_dir(dft_path)
    print(dataset_filepath_list)
    eval_filepath_list = []
    # 过滤用于评估的轨迹
    for i, filepath in enumerate(dataset_filepath_list):
        # if 'phone1' in filepath and 'user8' in filepath:
        #     eval_filepath_list.append(filepath)
        # if 'phone5' in filepath and 'user15' in filepath:
        #     eval_filepath_list.append(filepath)
        # if 'phone1' in filepath and 'user4' in filepath:
        #     eval_filepath_list.append(filepath)
        eval_filepath_list.append(filepath)
        if 'phone1' in filepath and 'user8' in filepath and pose == 'flat':
            eval_filepath_list.append(filepath)
        if 'phone1' in filepath and 'user5' in filepath and (pose == 'calling' or pose == 'pocket'):
            eval_filepath_list.append(filepath)
    return sorted(eval_filepath_list)


def parse_sen_from_file(filepath):
    ts_sen_map = {}
    fd = open(filepath, 'r')
    for i, line in enumerate(fd):
        if i == 0:
            continue
        sen_unit = SenUnitV2(line)
        # print(sen_unit.ts)
        if sen_unit.ts not in ts_sen_map:
            ts_sen_map[sen_unit.ts] = [sen_unit]
        else:
            ts_sen_map[sen_unit.ts].append(sen_unit)
    fd.close()
    return ts_sen_map


def joint_sen_unit2ftr_str_v2(ts_sen_map, pose='flat', version=None):
    pose_idx_map = {'flat': 1, 'calling': 2, 'pocket': 3}
    # print(ts_sen_map)
    ts_list = sorted(ts_sen_map.keys())
    # print(ts_list)
    ts_list = ts_list[3:-3]  # 过滤采样时间段前后三秒
    # print(ts_list)
    crnt_ts_list, next_ts_list = ts_list[:-1], ts_list[1:]
    # print(crnt_ts_list, next_ts_list)
    sen_pair_list = []
    for crnt_ts, next_ts in zip(crnt_ts_list, next_ts_list):
        # print("ts: ", crnt_ts, next_ts)
        if next_ts - crnt_ts > 1:
            continue
        # if 40 <= len(ts_sen_map[crnt_ts]) <= 60:
        #     sen_pair_list.append((ts_sen_map[crnt_ts], ts_sen_map[next_ts]))
        sen_pair_list.append((ts_sen_map[crnt_ts], ts_sen_map[next_ts]))
    # print(sen_pair_list[0][0][0].acc_array)
    print(len(sen_pair_list))

    sample_str_list, ori_gy_str_list = [], []
    ts_ori_gy_str_map = {}
    for crnt_sen_unit_list, next_sen_unit_list in sen_pair_list:
        # print(len(crnt_sen_unit_list))
        crnt_sen_unit = crnt_sen_unit_list[0]
        # print(crnt_sen_unit.acc_array)
        crnt_lon, crnt_lat, crnt_spd, crnt_brng = crnt_sen_unit.lon, crnt_sen_unit.lat, \
                                                  crnt_sen_unit.speed, crnt_sen_unit.abs_bearing
        next_sen_unit = next_sen_unit_list[0]
        next_lon, next_lat, next_spd, next_brng = next_sen_unit.lon, next_sen_unit.lat, \
                                                  next_sen_unit.speed, next_sen_unit.abs_bearing
        # if geo_util.haversine_formula(crnt_lon, crnt_lat, next_lon, next_lat) > 3.0:
        #     print('1234')
        #     continue
        # crnt_spd, next_spd = ComTools.rewrite_spd(crnt_lon, crnt_lat, crnt_spd, next_lon, next_lat,
        #                                           next_spd, phone_pose=phone_pose)

        # if len(crnt_sen_unit_list) < 50:
        #     crnt_sen_unit_list.extend(copy.deepcopy([crnt_sen_unit_list[-1]] *
        #                                             (50 - len(crnt_sen_unit_list))))
        # else:
        #     for i in range(len(crnt_sen_unit_list) - 50):
        #         crnt_sen_unit_list.pop(-1)
        # print(len(crnt_sen_unit_list))

        if version is None:
            # acc_ftr_array = [[u.accx, u.accy, u.accz] for u in crnt_sen_unit_list]
            # # print("array ", acc_ftr_array)
            # gy_ftr_array = [[u.gyrx, u.gyry, u.gyrz] for u in crnt_sen_unit_list]
            # acc_ftr_str = ','.join(['%.4f,%.4f,%.4f' % (x[0], x[1], x[2]) for x in acc_ftr_array])
            # # print("str ", acc_ftr_str)
            # gy_ftr_str = ','.join(['%.4f,%.4f,%.4f' % (x[0], x[1], x[2]) for x in gy_ftr_array])
            acc_ftr_array = [u.acc_array for u in crnt_sen_unit_list]
            # print(acc_ftr_array)
            gy_ftr_array = [u.gyro_array for u in crnt_sen_unit_list]
            acc_ftr_str = ','.join(acc_ftr_array[0])
            # print(acc_ftr_str)
            gy_ftr_str = ','.join(gy_ftr_array[0])
        else:
            rnn_ftr_gen = RnnFtrGenerator(phone_pose=pose, ftr_ver_list=['v1.0', 'v2.0'])
            ftr_list = rnn_ftr_gen.calc_sec_ftr_by_sen_data(crnt_sen_unit_list, version=version)
            acc_ftr_list, gy_ftr_list = [], []
            for k, ftr in enumerate(ftr_list):
                if 'acc' in rnn_ftr_gen.ftr_version_map[version]['idx_name_map'][k]:
                    acc_ftr_list.append(ftr)
                elif 'gy' in rnn_ftr_gen.ftr_version_map[version]['idx_name_map'][k]:
                    gy_ftr_list.append(ftr)
                else:
                    pass
            acc_ftr_str = ','.join(['%.4f' % x for x in acc_ftr_list])
            gy_ftr_str = ','.join(['%.4f' % x for x in gy_ftr_list])

        ts = crnt_sen_unit_list[0].ts
        sample_str = '%d,%d,%.6f,%.6f,%.4f,%.4f,%.6f,%.6f,%.4f,%.4f,%s,%s' % \
                     (ts, pose_idx_map[pose],
                      crnt_lon, crnt_lat, crnt_spd, crnt_brng,
                      next_lon, next_lat, next_spd, next_brng,
                      acc_ftr_str, gy_ftr_str)
        sample_str_list.append(sample_str)

        # 保留陀螺仪原始数据, 用于后续积分
        # ori_gy_arr = [[u.gyrx, u.gyry, u.gyrz] for u in crnt_sen_unit_list]
        ori_gy_arr = [u.gyro_array for u in crnt_sen_unit_list]
        ori_gy_str = ','.join(ori_gy_arr[0])
        # ori_gy_str = ','.join(['%.4f,%.4f,%.4f' % (x[0], x[1], x[2]) for x in ori_gy_arr])
        ori_gy_str_list.append('%d,%s' % (ts, ori_gy_str))
        ts_ori_gy_str_map[ts] = ori_gy_str
        # gy_ftr_array = [u.gyro_array for u in crnt_sen_unit_list]
        # acc_ftr_str = ','.join(acc_ftr_array[0])
        # # print(acc_ftr_str)
        # gy_ftr_str = ','.join(gy_ftr_array[0])

    print('sample_str_cnt=%d' % len(sample_str_list))
    return sample_str_list, ts_ori_gy_str_map


def parse_str2dict(line, version=None):
    _items = line.strip().split(',')
    ts = int(_items[0])
    _items = [float(x) for x in _items[1:]]

    pose_idx = _items[0]
    crnt_lon, crnt_lat, crnt_spd, crnt_brng = _items[1:5]
    crnt_bp = BasicPoint(ts, lon=crnt_lon, lat=crnt_lat, speed=crnt_spd, bearing=crnt_brng)

    next_lon, next_lat, next_spd, next_brng = _items[5:9]
    next_bp = BasicPoint(ts + 1, lon=next_lon, lat=next_lat, speed=next_spd, bearing=next_brng)

    if version is None:
        acc_ftr_list, gy_ftr_list = _items[9:759], _items[759:1509]
    elif version == 'v1.0':
        acc_ftr_list, gy_ftr_list = _items[9:21], _items[21:33]
    elif version == 'v2.0':
        acc_ftr_list, gy_ftr_list = _items[9:39], _items[39:69]
    else:
        raise ValueError('Feature version illegal!')

    return {ts: [pose_idx, crnt_bp, (acc_ftr_list, gy_ftr_list), next_bp]}


def intgr_gyr(gyr_axis_list):
    """ 陀螺仪角速度积分 """
    gyr_brng_diff_list = []
    for i in range(0, len(gyr_axis_list), 50):
        sec_gyr_axis_list = gyr_axis_list[i:i + 50]
        brng_diff = sum(sec_gyr_axis_list)
        gyr_brng_diff_list.append(brng_diff * -1.0)
    return gyr_brng_diff_list


def intgr_gyr_v2(gyr_axis_list):
    """ 陀螺仪积分 version2 """
    gyr_brng_diff_list = []
    for i in range(0, len(gyr_axis_list), 50):
        sec_gyr_axis_list = gyr_axis_list[i:i + 50]
        brng_diff = sum([x * 0.02 for x in sec_gyr_axis_list]) * 180.0 / math.pi
        gyr_brng_diff_list.append(brng_diff * -1.0)
    return gyr_brng_diff_list


def calc_brng_change(pred_brng, gt_brng):
    brng_diff = min(abs(pred_brng - gt_brng), 360 - abs(pred_brng - gt_brng))
    if round(gt_brng + brng_diff, 2) == round(pred_brng, 2):
        is_clockwise = True  # 相对真值顺时针偏航
    else:
        is_clockwise = False  # 相对真值逆时针偏航
    return brng_diff * (2 * is_clockwise - 1)


if __name__ == '__main__':
    phone_pose = sys.argv[1]
    TRAIN_SEQ_LEN = int(sys.argv[2])
    if len(sys.argv) > 3:
        EVAL_SEQ_LEN = int(sys.argv[3])
    else:
        EVAL_SEQ_LEN = TRAIN_SEQ_LEN
    ftr_version = None
    if len(sys.argv) > 4:
        ftr_version = sys.argv[4]
    print('phone_pose = %s, TRAIN_SEQ_LEN = %d, EVAL_SEQ_LEN = %d, ftr_version = %s' \
          % (phone_pose, TRAIN_SEQ_LEN, EVAL_SEQ_LEN, ftr_version))

    pose_path_map = {'flat': './dataset/user8',
                     'calling': '/Users/zhushuli/Desktop/PDR_dataset/220503_Wground_recovery/calling',
                     'pocket': '/Users/zhushuli/Desktop/PDR_dataset/220503_Wground_recovery/pocket'}
    dataset_file_path_list = list_eval_files(dft_path=pose_path_map[phone_pose], pose=phone_pose)
    # dataset_file_path_list = ['./dataset/lstm_eval_dataset_flat.txt']
    print('abc ', dataset_file_path_list)
    for i, filepath in enumerate(dataset_file_path_list):
        print('%02d: %s' % (i, filepath))

    # weight_name = 'spdLSTM_weight_%s_seqLen%d_%s3' % (phone_pose, TRAIN_SEQ_LEN, ftr_version)
    weight_name = 'model17.pt'
    weight_path = './model/%s' % weight_name
    model = load_rnn_net(version=ftr_version, trained_model=weight_path)
    model.eval()

    spd_sum_err_list, final_dist_err_list = [], []
    gross_spd_err_list = []
    pred_speed = []
    gt_speed = []
    for idx, filepath in enumerate(dataset_file_path_list):
        print('Read %s' % filepath)
        ts_sen_map = parse_sen_from_file(filepath)
        sample_str_list, ts_ori_gy_str_map = joint_sen_unit2ftr_str_v2(ts_sen_map,
                                                                       pose=phone_pose,
                                                                       version=ftr_version)

        ts_sec_ftr_map = {}
        for line in sample_str_list:
            ts_sec_ftr_map.update(parse_str2dict(line, version=ftr_version))
        ts_list = sorted(ts_sec_ftr_map.keys())

        sample_ftr_arr = []
        last_ts = None
        for i in range(0, len(ts_list) - 0):  # 可调参
            if len(sample_ftr_arr) == 0 or i == 0:
                sample_ftr_arr.append((ts_sec_ftr_map[ts_list[i]], ts_ori_gy_str_map[ts_list[i]]))
                last_ts = ts_list[i]
                continue
            if last_ts and ts_list[i] - last_ts <= 5:
                sample_ftr_arr.append((ts_sec_ftr_map[ts_list[i]], ts_ori_gy_str_map[ts_list[i]]))
                last_ts = ts_list[i]
            else:
                print('sample_ftr_array_cnt=%d' % len(sample_ftr_arr))
                sample_ftr_arr = [(ts_sec_ftr_map[ts_list[i]], ts_ori_gy_str_map[ts_list[i]])]
                last_ts = ts_list[i]

        base_bp = None  # 初始点
        gt_bp_list = []  # 不包含初始点的序列真值
        pred_spd_list, pred_brng_list = [], []
        print('sample_ftr_arr ', len(sample_ftr_arr))
        for i in range(0, len(sample_ftr_arr), EVAL_SEQ_LEN):
            seq_gyrx_list, seq_gyry_list, seq_gyrz_list = [], [], []
            init_spd, init_brng = None, None
            acc_ftr_arr, gy_ftr_arr, gamerv_ftr_arr = [], [], []

            seq_ftr_arr = sample_ftr_arr[i:i + EVAL_SEQ_LEN]
            # print('seq_ftr_arr ', len(seq_ftr_arr[0]))
            # print(seq_ftr_arr[0])
            for j, ((pose_idx, crnt_bp, ftr_tuple, next_bp), ori_gy_str) in enumerate(seq_ftr_arr):
                # 惯性数据特征
                acc_ftr_list, gy_ftr_list = ftr_tuple
                # 陀螺仪原始数据
                ori_gy_list = list(map(float, ori_gy_str.strip().split(',')))
                seq_gyrx_list.extend(ori_gy_list[0::3])
                seq_gyry_list.extend(ori_gy_list[1::3])
                seq_gyrz_list.extend(ori_gy_list[2::3])

                gt_bp_list.append(next_bp)
                if i == 0 and j == 0:
                    base_bp = crnt_bp
                    init_spd, init_brng = crnt_bp.speed, crnt_bp.abs_bearing
                elif j == 0:
                    init_spd, init_brng = pred_spd_list[-1], pred_brng_list[-1]

                acc_ftr_arr.append(acc_ftr_list)
                # print(acc_ftr_list)
                gy_ftr_arr.append(gy_ftr_list)
            acc_tensor = torch.tensor([acc_ftr_arr]).permute(0, 2, 1)
            # print('acc_tensor ', acc_tensor.shape)
            gy_tensor = torch.tensor([gy_ftr_arr]).permute(0, 2, 1)

            # 时序模型推算速度
            init_spd_tensor = torch.tensor([[[init_spd]]]).repeat(1, 1, len(seq_ftr_arr))
            pose_idx_tensor = torch.tensor([[[pose_idx]]]).repeat(1, 1, len(seq_ftr_arr))
            # print('123 ', acc_tensor)
            # print(acc_tensor.shape)
            pred = model(acc_tensor, gy_tensor, init_spd=init_spd_tensor, pose=None)
            # print("pred diff ", pred)
            pred_spd_diff_list = pred.squeeze().detach().numpy().tolist()
            if not isinstance(pred_spd_diff_list, list):
                pred_spd_diff_list = [pred_spd_diff_list]
            for pred_spd_diff in pred_spd_diff_list:
                # pred_spd_diff = 0.0
                # pred_spd = max(0.0, min(2.0, init_spd + pred_spd_diff))
                pred_spd = max(0.0, init_spd + pred_spd_diff)
                pred_spd_list.append(pred_spd)

            # 角度直接使用积分结果
            gyr_brng_diff_list = intgr_gyr_v2(seq_gyrz_list)
            for gyr_brng_diff in gyr_brng_diff_list:
                if i == 0 and len(pred_brng_list) == 0:
                    gyr_brng = (init_brng + gyr_brng_diff) % 360
                else:
                    gyr_brng = (pred_brng_list[-1] + gyr_brng_diff) % 360
                pred_brng_list.append(gyr_brng)

        pred_bp_list = []
        spd_err_list, brng_err_list, dist_err_list = [], [], []
        for pred_spd, pred_brng, gt_bp in zip(pred_spd_list, pred_brng_list, gt_bp_list):
            if len(pred_bp_list) == 0:
                pred_lon, pred_lat = geo_util.position_from_angle_and_dist(base_bp.lon, base_bp.lat,
                                                                           pred_brng, pred_spd)
            else:
                pred_lon, pred_lat = geo_util.position_from_angle_and_dist(pred_bp_list[-1].lon, pred_bp_list[-1].lat,
                                                                           pred_brng, pred_spd)
            pred_bp = BasicPoint(gt_bp.ts, lon=pred_lon, lat=pred_lat, speed=pred_spd, bearing=pred_brng)
            pred_bp_list.append(pred_bp)
            # print('pred ', pred_bp.speed, gt_bp.speed)
            pred_speed.append(pred_bp.speed)
            gt_speed.append(gt_bp.speed)
            spd_err_list.append(pred_spd - gt_bp.speed)  # 每秒速度误差
            brng_err_list.append(calc_brng_change(pred_brng, gt_bp.abs_bearing))  # 每秒航向误差
            dist_err_list.append(geo_util.distance(pred_lon, pred_lat, gt_bp.lon, gt_bp.lat))
            # print(pred_lon, pred_lat, gt_bp.lon, gt_bp.lat)

        gross_spd_err_list.extend(spd_err_list)

        print('-' * 17 + ' Speed (m/s) ' + '-' * 17)
        print('Speed absolute error sum = %.2f' % sum([abs(x) for x in spd_err_list]))
        print('Speed absolute error per second = %.2f' % np.mean([abs(x) for x in spd_err_list]))
        print('Speed second min = %.2f, max = %.2f' % (min([abs(x) for x in spd_err_list]),
                                                       max([abs(x) for x in spd_err_list])))
        print('-' * 17 + ' Bearing (degree)' + '-' * 17)
        print('Bearing absolute error sum = %.2f' % sum([abs(x) for x in brng_err_list]))
        print('Bearing absolute error per second = %.2f' % np.mean([abs(x) for x in brng_err_list]))
        print('-' * 17 + ' Distance (m)' + '-' * 17)
        print('Distance absolute error sum = %.2f' % sum([abs(x) for x in dist_err_list]))
        print('Distance absolute error per second = %.2f' % np.mean([abs(x) for x in dist_err_list]))
        print('Distance second min = %.2f, max = %.2f' % (min([abs(x) for x in dist_err_list]),
                                                          max([abs(x) for x in dist_err_list])))
        print('Final distance error = %.2f' % dist_err_list[-1])
        final_dist_err_list.append(dist_err_list[-1])

        walk_dist_list = []
        for i in range(len(gt_bp_list)):
            if i == 0:
                walk_dist_list.append(geo_util.distance(gt_bp_list[i].lon, gt_bp_list[i].lat,
                                                        base_bp.lon, base_bp.lat))
            else:
                walk_dist_list.append(geo_util.distance(gt_bp_list[i].lon, gt_bp_list[i].lat,
                                                        gt_bp_list[i - 1].lon, gt_bp_list[i - 1].lat))
        print('Walk distance = %.2f' % sum(walk_dist_list), end=', ')

        pred_dist_list = []
        for i in range(len(pred_bp_list)):
            if i == 0:
                pred_dist_list.append(geo_util.distance(pred_bp_list[i].lon, pred_bp_list[i].lat,
                                                        base_bp.lon, base_bp.lat))
            else:
                pred_dist_list.append(geo_util.distance(pred_bp_list[i].lon, pred_bp_list[i].lat,
                                                        pred_bp_list[i - 1].lon, pred_bp_list[i - 1].lat))
        print('Prediction distance = %.2f' % sum(pred_dist_list))

        # gt_trace_str = '%s;%s' % (base_bp.lon_lat_str_gcj02, ';'.join([u.lon_lat_str_gcj02 for u in gt_bp_list]))
        # pred_trace_str = '%s;%s' % (base_bp.lon_lat_str_gcj02, ';'.join([u.lon_lat_str_gcj02 for u in pred_bp_list]))
        # print('%s\t%s' % (gt_trace_str, pred_trace_str))

        # 轨迹可视化

        # 速度可视化
        # fig, ax = plt.subplots(figsize=(9, 6))
        # ax.plot([base_bp.speed] + pred_spd_list, label='pred', linewidth=2.5, color='DarkGreen')
        # ax.plot([base_bp.speed] + [u.speed for u in gt_bp_list], label='gt', linewidth=2.5, color='Red')
        # ax.set_xlabel('Time (s)', fontsize=20)
        # ax.set_ylabel('Speed (m/s)', fontsize=20)
        # ax.set_title(filepath, fontsize=8)
        # plt.ylim(0.5, 1.5)
        # plt.legend(fontsize=20)
        # plt.grid(alpha=0.25)
        # plt.tight_layout()
        # plt.savefig('./images/tmp/demo_speed_idx%d.png' % idx, dpi=300)
        # plt.show()

        # 角度可视化
        # fig, ax = plt.subplots(figsize=(9, 6))
        # ax.plot([base_bp.abs_bearing] + pred_brng_list, label='pred', linewidth=2.5, color='DarkGreen')
        # ax.plot([base_bp.abs_bearing] + [u.abs_bearing for u in gt_bp_list], label='gt', linewidth=2.5, color='Red')
        # ax.set_xlabel('Time (s)', fontsize=20)
        # ax.set_ylabel('Bearing (m/s)', fontsize=20)
        # ax.set_title(filepath, fontsize=8)
        # plt.legend(fontsize=20)
        # plt.grid(alpha=0.25)
        # plt.tight_layout()
        # plt.savefig('./images/tmp/demo_bearing_idx%d.png' % idx, dpi=300)
        # plt.show()

        spd_sum_err_list.append(sum([abs(x) for x in spd_err_list]))

    # 每秒速度误差写入文件
    with open('./tmp_csv/%s_spdErr.csv' % weight_name, 'w') as fd:
        for spd_err in gross_spd_err_list:
            fd.write('%.2f\n' % spd_err)

    # spd_sum_err_list = spd_sum_err_list[:-1]
    # final_dist_err_list = final_dist_err_list[:-1]
    print([round(x, 2) for x in spd_sum_err_list])
    print('Mean = %.2f m/s, std = %.2f m/s' % (np.mean(spd_sum_err_list), np.std(spd_sum_err_list)))
    print([round(x, 2) for x in final_dist_err_list])
    print('Mean = %.2f m, std = %.2f m' % (np.mean(final_dist_err_list), np.std(final_dist_err_list)))

    print(pred_speed)
    print(gt_speed)
