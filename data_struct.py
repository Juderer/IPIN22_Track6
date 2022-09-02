#!/user/bin/env python
# -*- coding:utf-8 -*-
"""
@Time       : 2022/8/29 17:23
@Author     : zhushuli
@File       : data_struct.py
@DevTool    : PyCharm
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from coord_utils import *


class gnssUnit():
    def __init__(self, line, pre_gnss=None):
        assert isinstance(line, str) and line.startswith('GNSS')
        _items = line.strip().split(';')
        self.app_ts = float(_items[1])
        self.lat = float(_items[2])
        self.lng = float(_items[3])
        self.altitude = float(_items[4])
        self.bearing = float(_items[5])
        # self.accuracy = float(_items[6])
        # self.speed = float(_items[7])
        self.speed = float(_items[6])
        self.accuracy = float(_items[7])
        self.gps_tow = float(_items[8])
        self.satellite_in_view = int(_items[9])
        self.satellite_in_use = int(_items[10])
        self.pre_gnss = pre_gnss
        self.lng_gcj02, self.lat_gcj02 = geo_util.wgs84_to_gcj02(self.lng, self.lat)

    @property
    def loc_wgs84_str(self):
        return '%.6f,%.6f' % (self.lng, self.lat)

    @property
    def loc_gcj02_str(self):
        return '%.6f,%.6f' % (self.lng_gcj02, self.lat_gcj02)

    @property
    def loc_speed(self):
        # if self.pre_gnss is None:
        #     return -1
        # pre_spd = geo_util.distance(self.lng, self.lat, self.pre_gnss.lng, self.pre_gnss.lat)
        # return max(0.0, min(33.0, pre_spd))
        return self.speed

    def __repr__(self):
        return '%.3f,%.6f,%.6f,%.3f,%.3f' % (self.app_ts, self.lng, self.lat, self.bearing, self.speed)


class acceUnit():
    def __init__(self, line):
        assert isinstance(line, str) and line.startswith('ACCE')
        _items = line.strip().split(';')
        self.app_ts = float(_items[1])
        self.sensor_ts = float(_items[2])
        self.acc_x = float(_items[3])
        self.acc_y = float(_items[4])
        self.acc_z = float(_items[5])
        self.accuracy = int(_items[6])

    @property
    def sen_x(self):
        return self.acc_x

    @property
    def sen_y(self):
        return self.acc_y

    @property
    def sen_z(self):
        return self.acc_z

    def __repr__(self):
        return '%.3f,%.3f,%.3f,%.3f' % (self.app_ts, self.acc_x, self.acc_y, self.acc_z)


class gyroUnit():
    def __init__(self, line):
        assert isinstance(line, str) and line.startswith('GYRO')
        _items = line.strip().split(';')
        self.app_ts = float(_items[1])
        self.sensor_ts = float(_items[2])
        self.gyr_x = float(_items[3]) * 180.0 / np.pi
        self.gyr_y = float(_items[4]) * 180.0 / np.pi
        self.gyr_z = float(_items[5]) * 180.0 / np.pi
        self.accuracy = int(_items[6])

    @property
    def sen_x(self):
        return self.gyr_x

    @property
    def sen_y(self):
        return self.gyr_y

    @property
    def sen_z(self):
        return self.gyr_z

    def __repr__(self):
        return '%.3f,%.3f,%.3f,%.3f' % (self.app_ts, self.gyr_x, self.gyr_y, self.gyr_z)


class ahrsUnit():
    def __init__(self, line):
        assert isinstance(line, str) and line.startswith('AHRS')
        _items = line.strip().split(';')
        self.app_ts = float(_items[1])
        self.sensor_ts = float(_items[2])
        self.pitch_x = float(_items[3])
        self.roll_y = float(_items[4])
        self.yaw_z = float(_items[5])
        self.rot_vec_x = float(_items[6])
        self.rot_vec_y = float(_items[7])
        self.rot_vec_z = float(_items[8])
        self.accuracy = int(_items[9])

    def __repr__(self):
        return '%.3f,%.3f,%.3f,%.3f' % (self.app_ts, self.pitch_x, self.roll_y, self.yaw_z)


class posiUnit():
    def __init__(self, line):
        assert isinstance(line, str) and line.startswith('POSI')
        _items = line.split(';')
        self.gps_tow = float(_items[1])
        self.lat = float(_items[2])
        self.lng = float(_items[3])
        self.altitude = float(_items[4])
        self.lng_gcj02, self.lat_gcj02 = geo_util.wgs84_to_gcj02(self.lng, self.lat)

    @property
    def loc_wgs84_str(self):
        return '%.6f,%.6f' % (self.lng, self.lat)

    @property
    def loc_gcj02_str(self):
        return '%.6f,%.6f' % (self.lng_gcj02, self.lat_gcj02)

    def __repr__(self):
        return '%.3f,%.6f,%.6f' % (self.gps_tow, self.lng, self.lat)
