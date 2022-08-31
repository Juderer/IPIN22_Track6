#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2022/4/13 14:52
@Author     : zhushuli
@DevTool    : PyCharm
@File       : lstm_ftr_produce_v2.py
@CopyFrom   : lstm_ftr_produce.py
"""
import os
import sys
import copy
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from data_struct import *
from coord_utils import *
from utils import *
from ftr_generator import *


def list_dataset_files(dft_path='./dataset/220407_Wground_recovery', POSE=None):
    """
    Args
        dft_path: 数据集所在目录
        dataset_type: 数据集类型(train\test)
    Return
        文件列表
    """
    dataset_filepath_list = ComTools.traverse_dir(dft_path)
    filtered_filepath_list = []
    # 过滤用于评估的轨迹
    for i, filepath in enumerate(dataset_filepath_list):
        # if 'phone1' in filepath and 'user8' in filepath:
        #     continue
        # if 'phone5' in filepath and 'user15' in filepath:
        #     continue
        if 'phone1' in filepath and 'user8' in filepath and POSE and POSE == 'flat':
            continue
        if 'phone5' in filepath and 'user14' in filepath and POSE and POSE == 'flat':  # 数据有问题
            continue
        if 'phone1' in filepath and 'user5' in filepath and POSE and (POSE == 'calling' or POSE == 'pocket'):
            continue
        filtered_filepath_list.append(filepath)
    random.shuffle(filtered_filepath_list)

    train_filepath_list, test_filepath_list = [], []
    for i, filepath in enumerate(filtered_filepath_list, 1):
        if i % 4 == 0:
            test_filepath_list.append(filepath)
        else:
            train_filepath_list.append(filepath)

    # train_filepath_list = filtered_filepath_list[:len(filtered_filepath_list) // 4 * 3]
    # test_filepath_list = filtered_filepath_list[len(filtered_filepath_list) // 4 * 3:]
    return {'train': train_filepath_list, 'test': test_filepath_list}


def parse_sen_from_file(filepath):
    ts_sen_map = {}
    fd = open(filepath, 'r')
    for i, line in enumerate(fd):
        if i == 0: continue
        sen_unit = SenUnitV2(line)
        if sen_unit.ts not in ts_sen_map:
            ts_sen_map[sen_unit.ts] = [sen_unit]
        else:
            ts_sen_map[sen_unit.ts].append(sen_unit)
    fd.close()
    return ts_sen_map


def joint_sen_unit2ftr_str_v2(ts_sen_map, SEQ_LEN=1, STRIDE=1, REJECT_SEC=3, POSE='flat', VERSION=None, ftr_hz=50):
    """
    Args
        ts_sen_map: 秒级时间戳与对应传感器数据的映射
        SEQ_LEN: 构建特征的序列长度, (>=1)
        STRIDE: 滑动窗口长度(步长), (>=1)
        REJECT_SEC: 除去采集数据时点击按钮带来的噪声数据
        POSE: 手机姿态(flat, calling, pocket)
        VERSION: 特征版本号, 若为None表示使用原始数据
    Return
        特征列表
    """
    pose_idx_map = {'flat': 1, 'calling': 2, 'pocket': 3}

    ts_list = sorted(ts_sen_map.keys())
    ts_list = ts_list[REJECT_SEC:-REJECT_SEC]  # 过滤采样时间段前后

    sample_str_list = []
    for i in range(0, len(ts_list) - SEQ_LEN, STRIDE):
        sample_ts_list = ts_list[i:i + SEQ_LEN + 1]

        sec_str_list = []
        for j in range(0, len(sample_ts_list) - 1, 1):
            crnt_sen_unit_list, next_sen_unit_list = ts_sen_map[sample_ts_list[j]], \
                                                     ts_sen_map[sample_ts_list[j + 1]]
            crnt_head_sen_unit = crnt_sen_unit_list[0]
            crnt_lon, crnt_lat, crnt_spd, crnt_brng = crnt_head_sen_unit.lon, crnt_head_sen_unit.lat, \
                                                      crnt_head_sen_unit.speed, crnt_head_sen_unit.bearing
            next_head_sen_unit = next_sen_unit_list[0]
            next_lon, next_lat, next_spd, next_brng = next_head_sen_unit.lon, next_head_sen_unit.lat, \
                                                      next_head_sen_unit.speed, next_head_sen_unit.bearing

            if geo_util.haversine_formula(crnt_lon, crnt_lat, next_lon, next_lat) > 3.0:
                continue
            if 40 <= len(crnt_sen_unit_list) <= 50:
                crnt_sen_unit_list.extend(copy.deepcopy([crnt_sen_unit_list[-1]] *
                                                        (50 - len(crnt_sen_unit_list))))
            elif 50 < len(crnt_sen_unit_list) <= 60:
                for k in range(len(crnt_sen_unit_list) - 50):
                    crnt_sen_unit_list.pop(-1)
            else:
                continue

            if VERSION is None:
                acc_ftr_arr = [[u.accx, u.accy, u.accz] for u in crnt_sen_unit_list]
                gy_ftr_arr = [[u.gyrx, u.gyry, u.gyrz] for u in crnt_sen_unit_list]
                acc_ftr_str = ','.join(['%.4f,%.4f,%.4f' % (x[0], x[1], x[2]) for x in acc_ftr_arr])
                gy_ftr_str = ','.join(['%.4f,%.4f,%.4f' % (x[0], x[1], x[2]) for x in gy_ftr_arr])
                grv_ftr_arr = [[u.gamervx, u.gamervy, u.gamervz, u.gamervs] for u in crnt_sen_unit_list]
                grv_ftr_str = ','.join(['%.4f,%.4f,%.4f,%.4f' % (x[0], x[1], x[2], x[3]) for x in grv_ftr_arr])
            else:
                ftr_list = rnn_ftr_gen.calc_sec_ftr_by_sen_data(crnt_sen_unit_list, version=VERSION, ftr_hz=ftr_hz)
                acc_ftr_list, gy_ftr_list, grv_ftr_list = [], [], []
                for k, ftr in enumerate(ftr_list):
                    if 'acc' in rnn_ftr_gen.ftr_version_map[VERSION]['idx_name_map'][k]:
                        acc_ftr_list.extend(ftr)
                    elif 'gy' in rnn_ftr_gen.ftr_version_map[VERSION]['idx_name_map'][k]:
                        gy_ftr_list.extend(ftr)
                    elif 'grv' in rnn_ftr_gen.ftr_version_map[VERSION]['idx_name_map'][k]:
                        grv_ftr_list.extend(ftr)
                    else:
                        pass
                acc_ftr_str = ','.join(['%.4f' % x for x in acc_ftr_list])
                gy_ftr_str = ','.join(['%.4f' % x for x in gy_ftr_list])

            sec_str = '%d,%.6f,%.6f,%.4f,%.4f,%.6f,%.6f,%.4f,%.4f,%s,%s' \
                      % (pose_idx_map[POSE],
                         crnt_lon, crnt_lat, crnt_spd, crnt_brng,
                         next_lon, next_lat, next_spd, next_brng,
                         acc_ftr_str, gy_ftr_str)
            sec_str_list.append(sec_str)

        if len(sec_str_list) == SEQ_LEN:
            sample_str_list.append('|'.join(sec_str_list))
    print('sample_str_cnt=%d' % len(sample_str_list))
    return sample_str_list


if __name__ == '__main__':
    phone_pose = sys.argv[1]  # flat, calling, pocket
    seq_len = int(sys.argv[2])
    stride = int(sys.argv[3])
    reject_sec = int(sys.argv[4])
    ftr_version = None
    level = None
    ftr_hz = 50
    if len(sys.argv) > 5:
        ftr_version = sys.argv[5]
    if len(sys.argv) > 6:
        ftr_hz = int(sys.argv[6])
    print('phone_pose = %s, seq_len = %d, stride = %d, ftr_version = %s, ftr_hz = %s' \
          % (phone_pose, seq_len, stride, ftr_version, ftr_hz))

    rnn_ftr_gen = RnnFtrGenerator(phone_pose=phone_pose, ftr_ver_list=['v1.0'], ftr_hz=ftr_hz)

    # flat: /Users/zhushuli/Desktop/PDR_dataset/220407_Wground_recovery or 220407_Wground_recovery_align_v2
    # individuation flat: /Users/zhushuli/Desktop/PDR_dataset/indvd/phone1_2
    pose_path_map = {'flat': ['/Users/zhushuli/Desktop/PDR_dataset/220407_Wground_recovery', ],
                     'calling': ['/Users/zhushuli/Desktop/PDR_dataset/220503_Wground_recovery/calling',
                                 '/Users/zhushuli/Desktop/PDR_dataset/220606_Wground_zParking_recovery/Wground/calling'],
                     'pocket': ['/Users/zhushuli/Desktop/PDR_dataset/220503_Wground_recovery/pocket',
                                '/Users/zhushuli/Desktop/PDR_dataset/220606_Wground_zParking_recovery/Wground/pocket']}
    sample_str_map = {}
    for _pose, _path_list in pose_path_map.items():
        if phone_pose != 'all' and _pose != phone_pose:
            continue
        for _path in _path_list:
            filepath_map = list_dataset_files(dft_path=_path, POSE=_pose)
            for dataset_type, filepath_list in filepath_map.items():
                print('-' * 24 + dataset_type + '-' * 24)
                sample_str_list = []
                for filepath in filepath_list:
                    print('read %s' % filepath)
                    ts_sen_map = parse_sen_from_file(filepath)
                    sample_str_list.extend(joint_sen_unit2ftr_str_v2(ts_sen_map,
                                                                     SEQ_LEN=seq_len,
                                                                     STRIDE=stride,
                                                                     REJECT_SEC=reject_sec,
                                                                     POSE=_pose,
                                                                     VERSION=ftr_version,
                                                                     ftr_hz=ftr_hz))
                print('sample_str_cnt=%d' % len(sample_str_list))
                if dataset_type in sample_str_map:
                    sample_str_map[dataset_type].extend(sample_str_list)
                else:
                    sample_str_map[dataset_type] = sample_str_list

    for dataset_type, sample_str_list in sample_str_map.items():
        dataset_name = 'lstm_datasetv2_%s_%s_seqLen%d_%s_ftrHz%d.txt' \
                       % (dataset_type, phone_pose, seq_len, ftr_version, ftr_hz)
        if level is not None:
            dataset_name = 'lstm_datasetv2_%s_%s_seqLen%d_%s_level%d.txt' \
                           % (dataset_type, phone_pose, seq_len, ftr_version, level)
        print('Dataset Name: %s' % dataset_name)
        with open('./tmp_dataset/%s' % dataset_name, 'w') as fd:
            for i, sample in enumerate(sample_str_list):
                if level is not None and i == len(sample_str_list) * level // 100:
                    fd.write(sample)
                    break
                else:
                    fd.write('%s\n' % sample)

    # rnn_ftr_gen.calc_ftr_domain(version=ftr_version)

    print('Finish')
