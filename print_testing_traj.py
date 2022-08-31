#!/user/bin/env python
# -*- coding:utf-8 -*-
"""
@Time       : 2022/8/29 17:59
@Author     : zhushuli
@File       : print_testing_traj.py
@DevTool    : PyCharm
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_struct import *

if __name__ == '__main__':
    with open('./trials/IPIN2022_T7_TestingTrial02.txt', 'r') as fd:
        gnss_unit_list, posi_unit_list = [], []
        for i, line in enumerate(fd):
            if line.startswith('GNSS'):
                gnss_unit = gnssUnit(line)
                gnss_unit_list.append(gnss_unit)
            if line.startswith('POSI'):
                posi_unit = posiUnit(line)
                posi_unit_list.append(posi_unit)
    print('GNSS count: %d' % len(gnss_unit_list))
    print('POSI count: %d' % len(posi_unit_list))

    app_ts_list = [u.app_ts for u in gnss_unit_list]
    print(max(app_ts_list) - min(app_ts_list))

    with open('testing_gnss_traj.csv', 'w') as fd:
        start_app_ts = 0
        fd.write('lng,lat,bearing,speed,accuracy,appTs\n')
        for i, gnss_unit in enumerate(gnss_unit_list):
            if i == 0:
                fd.write('%s,%.2f,%.2f,%d,%.1f\n' % (gnss_unit.loc_gcj02_str, gnss_unit.bearing, gnss_unit.speed,
                                                     gnss_unit.accuracy, gnss_unit.app_ts))
                start_app_ts = gnss_unit.app_ts
            else:
                fd.write('%s,%.2f,%.2f,%d,%.1f\n' % (gnss_unit.loc_gcj02_str, gnss_unit.bearing, gnss_unit.speed,
                                                     gnss_unit.accuracy, gnss_unit.app_ts - start_app_ts))

    with open('testing_posi_traj.csv', 'w') as fd:
        start_gps_tow = 0
        fd.write('lng,lat,gpsTow\n')
        for i, posi_unit in enumerate(posi_unit_list):
            if i == 0:
                fd.write('%s,%.1f\n' % (posi_unit.loc_gcj02_str, posi_unit.gps_tow))
                start_gps_tow = posi_unit.gps_tow
            else:
                fd.write('%s,%.1f\n' % (posi_unit.loc_gcj02_str, posi_unit.gps_tow - start_gps_tow))

    print('Finish')
