#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2021/08/21 20:55
@Author     : Zhu Shuli
@File       : data_struct.py
@DevTool    : PyCharm
@Desc       : 常用数据结构
"""
import math
from coord_utils import *


class BasicPoint(object):
    def __init__(self, ts, lon=None, lat=None, speed=None, bearing=None):
        self.ts = ts
        self.lon = lon
        self.lat = lat
        self.speed = speed
        self.bearing = bearing

    def __repr__(self):
        return '%d,%.6f,%.6f,%.4f,%.4f' % (self.ts, self.lon, self.lat, self.speed, self.bearing)

    @property
    def abs_bearing(self):
        """ 若bearing中出现负值, 则: 向东为正, 向西为负 """
        if self.bearing >= 0.0:
            return self.bearing
        else:
            return 360.0 + self.bearing

    @property
    def lon_lat_str_wgs84(self):
        return '%.6f,%.6f' % (self.lon, self.lat)

    @property
    def lon_lat_str_gcj02(self):
        lon, lat = geo_util.wgs84_to_gcj02(self.lon, self.lat)
        return '%.6f,%.6f' % (lon, lat)

    def get_lon_lat_str(self, coord='wgs84'):
        if coord == 'gcj02':
            return self.lon_lat_str_gcj02
        return self.lon_lat_str_wgs84


class IndoorBasicPoint(object):
    def __init__(self, ts, x=None, y=None, speed=None, bearing=None):
        self.ts = ts
        if x is None or y is None or speed is None or bearing is None:
            self.x = self.y = self.speed = self.bearing = 0.0
        else:
            self.x = x
            self.y = y
            self.speed = speed
            self.bearing = bearing

    def __repr__(self):
        return '%d,%.4f,%.4f,%.4f,%.4f' % (self.ts, self.x, self.y, self.speed, self.bearing)


class SenUnit(object):
    def __init__(self, line):
        # string format:
        # sys_time,laccx,y,z,grax,y,z,gyrx,y,z,accx,y,z,magx,y,z,ori,lon,lat,speed,bearing,gps_time,step
        if 'null' in line:
            raise ValueError('sensor data is null.')
        _items = line.strip().split(',')
        self.ts_ms = int(_items[0])
        self.laccx = round(float(_items[1]), 4)
        self.laccy = round(float(_items[2]), 4)
        self.laccz = round(float(_items[3]), 4)
        self.gvtx = round(float(_items[4]), 4)
        self.gvty = round(float(_items[5]), 4)
        self.gvtz = round(float(_items[6]), 4)
        self.gyrx = round(float(_items[7]), 4)
        self.gyry = round(float(_items[8]), 4)
        self.gyrz = round(float(_items[9]), 4)
        self.accx = round(float(_items[10]), 4)
        self.accy = round(float(_items[11]), 4)
        self.accz = round(float(_items[12]), 4)
        self.magx = round(float(_items[13]), 4)
        self.magy = round(float(_items[14]), 4)
        self.magz = round(float(_items[15]), 4)
        self.ori = round(float(_items[16]), 4)
        self.lon = round(float(_items[17]), 6)
        self.lat = round(float(_items[18]), 6)
        self.speed = round(float(_items[19]), 4)
        self.bearing = round(float(_items[20]), 4)
        self.ts = int(_items[21]) // 1000
        if len(_items) > 22:
            self.step = round(float(_items[22]), 1)
        else:
            self.step = 0

    @property
    def timestamp(self):
        return self.ts_ms

    @property
    def out_str(self):
        return '%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,%.4f,%.4f,%d,%.1f' \
               % (self.ts_ms, self.laccx, self.laccy, self.laccz, self.gvtx, self.gvty, self.gvtz, self.gyrx, self.gyry, self.gyrz,
                  self.accx, self.accy, self.accz, self.magx, self.magy, self.magz, self.ori,
                  self.lon, self.lat, self.speed, self.bearing, self.ts * 1000, self.step)

    @property
    def item_info(self):
        return 'sys_time,laccx,y,z,grax,y,z,gyrx,y,z,accx,y,z,magx,y,z,ori,lon,lat,speed,bearing,gps_time,step'


class SenUnitV2(object):
    def __init__(self, line):
        """ string format:
        Sys_time,laccx,y,z,lacc_accu,grax,y,z,gra_accu,gyrx,y,z,gyr_accu,accx,y,z,acc_accu,
        magx,y,z,mag_accu,ori,rot_x,rot_y,rot_z,rot_s,rot_head_acc,rot_accu,
        grot_x,grot_y,grot_z,g_rot_s,g_rot_accu,lon,lat,speed,bearing,gps_time,step"""
        if 'null' in line:
            raise ValueError('sensor data (version 2.0) is null.')
        _items = line.strip().split(',')
        # print(_items)
        self.ts_ms = int(_items[0])
        self.lon = round(float(_items[1]), 6)
        self.lat = round(float(_items[2]), 6)
        self.speed = round(float(_items[3]), 4)
        self.bearing = round(float(_items[4]), 4)
        self.acc_array = _items[9:759]
        self.gyro_array = _items[759:1509]
        self.ts = int(_items[1509]) #// 1000
        # self.laccx = round(float(_items[1]), 4)
        # self.laccy = round(float(_items[2]), 4)
        # self.laccz = round(float(_items[3]), 4)
        # self.lacc_accu = round(float(_items[4]), 2)
        # self.gvtx = round(float(_items[5]), 4)
        # self.gvty = round(float(_items[6]), 4)
        # self.gvtz = round(float(_items[7]), 4)
        # self.gvt_accu = round(float(_items[8]), 2)
        # self.gyrx = round(float(_items[9]), 4)
        # self.gyry = round(float(_items[10]), 4)
        # self.gyrz = round(float(_items[11]), 4)
        # self.gyr_accu = round(float(_items[12]), 2)
        # self.accx = round(float(_items[13]), 4)
        # self.accy = round(float(_items[14]), 4)
        # self.accz = round(float(_items[15]), 4)
        # self.acc_accu = round(float(_items[16]), 2)
        # self.magx = round(float(_items[17]), 4)
        # self.magy = round(float(_items[18]), 4)
        # self.magz = round(float(_items[19]), 4)
        # self.mag_accu = round(float(_items[20]), 2)
        # self.ori = round(float(_items[21]), 4)
        # self.rvx = round(float(_items[22]), 4)
        # self.rvy = round(float(_items[23]), 4)
        # self.rvz = round(float(_items[24]), 4)
        # self.rvs = round(float(_items[25]), 4)
        # self.rv_head_acc = round(float(_items[26]), 2)
        # self.rv_accu = round(float(_items[27]), 2)
        # self.gamervx = round(float(_items[28]), 4)
        # self.gamervy = round(float(_items[29]), 4)
        # self.gamervz = round(float(_items[30]), 4)
        # self.gamervs = round(float(_items[31]), 4)
        # self.gamerv_accu = round(float(_items[32]), 2)
        # self.lon = round(float(_items[33]), 6)
        # self.lat = round(float(_items[34]), 6)
        # self.speed = round(float(_items[35]), 4)
        # self.bearing = round(float(_items[36]), 4)
        # self.ts = int(_items[37]) // 1000
        if len(_items) > 38:
            self.step = round(float(_items[38]), 1)
        else:
            self.step = 0

    @property
    def timestamp(self):
        return self.ts_ms

    @property
    def abs_bearing(self):
        """ 若bearing中出现负值, 则: 向东为正, 向西为负 """
        if self.bearing >= 0.0:
            return self.bearing
        else:
            return 360.0 + self.bearing

    @property
    def out_str(self):
        return '%d,%.4f,%.4f,%.4f,%.2f,%.4f,%.4f,%.4f,%.2f,%.4f,%.4f,%.4f,%.2f,%.4f,%.4f,%.4f,%.2f,%.4f,%.4f,%.4f,' \
               '%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%.2f,%.6f,%.6f,%.4f,%.4f,%d,%.1f' \
               % (self.ts_ms, self.laccx, self.laccy, self.laccz, self.lacc_accu,
                  self.gvtx, self.gvty, self.gvtz, self.gvt_accu, self.gyrx, self.gyry, self.gyrz, self.gyr_accu,
                  self.accx, self.accy, self.accz, self.acc_accu, self.magx, self.magy, self.magz, self.mag_accu,
                  self.ori, self.rvx, self.rvy, self.rvz, self.rvs, self.rv_head_acc, self.rv_accu,
                  self.gamervx, self.gamervy, self.gamervz, self.gamervs, self.gamerv_accu,
                  self.lon, self.lat, self.speed, self.bearing, self.ts * 1000, self.step)

    @property
    def to_list(self):
        return [self.ts_ms, self.laccx, self.laccy, self.laccz, self.lacc_accu,
                self.gvtx, self.gvty, self.gvtz, self.gvt_accu, self.gyrx, self.gyry, self.gyrz, self.gyr_accu,
                self.accx, self.accy, self.accz, self.acc_accu, self.magx, self.magy, self.magz, self.mag_accu,
                self.ori, self.rvx, self.rvy, self.rvz, self.rvs, self.rv_head_acc, self.rv_accu,
                self.gamervx, self.gamervy, self.gamervz, self.gamervs, self.gamerv_accu,
                self.lon, self.lat, self.speed, self.bearing, self.ts * 1000, self.step]

    def get_accuracy(self):
        return [self.lacc_accu, self.gvt_accu, self.gyr_accu, self.acc_accu, self.mag_accu,
                self.rv_head_acc, self.rv_accu, self.gamerv_accu]

    @property
    def item_info(self):
        return 'Sys_time,laccx,y,z,lacc_accu,grax,y,z,gra_accu,gyrx,y,z,gyr_accu,accx,y,z,acc_accu,' \
               'magx,y,z,mag_accu,ori,rv_x,rv_y,rv_z,rv_s,rv_head_acc,rv_accu,' \
               'gamerv_x,gamerv_y,gamerv_z,gamerv_s,gamerv_accu,lon,lat,speed,bearing,gps_time,step'

    def get_by_imu_axis(self, imu='acc', axis='x'):
        """
        Args
            imu: 加速度计与陀螺仪之一['acc', 'gy']
            axis: IMU某一轴['x', 'y', 'z']
        """
        if imu == 'acc':
            if axis == 'x': return self.accx
            elif axis == 'y': return self.accy
            elif axis == 'z': return self.accz
            else: raise ValueError("axis can only be 'x', 'y' or 'z'.")
        elif imu == 'gy':
            if axis == 'x': return self.gyrx
            elif axis == 'y': return self.gyry
            elif axis == 'z': return self.gyrz
            else: raise ValueError("axis can only be 'x', 'y' or 'z'.")
        else:
            raise ValueError("imu can only be 'acc' or 'gy'")


class IndoorSenUnitV2(SenUnitV2):
    def __init__(self, line):
        super(IndoorSenUnitV2, self).__init__(line)
        _items = line.strip().split(',')
        if len(_items[0]) == 13:
            self.ts = int(_items[0]) // 1000
        elif len(_items[0]) == 10:
            self.ts = int(_items[0])


class OxIODVicon(object):
    """
    牛津大学PDR数据集OxIOD;
    利用Vicon System采集的Ground Truth;
    参阅http://deepio.cs.ox.ac.uk/
    """
    def __init__(self, line):
        self.__FLOAT_ACC__ = 4  # 浮点数精度
        _items = line.strip().split(',')
        self.time = int(_items[0])
        self.header = int(_items[1])
        # 位移量(可以使用水平面位移量)
        self.trans_x = round(float(_items[2]), self.__FLOAT_ACC__)
        self.trans_y = round(float(_items[3]), self.__FLOAT_ACC__)
        self.trans_z = round(float(_items[4]), self.__FLOAT_ACC__)
        self.rota_x = round(float(_items[5]), self.__FLOAT_ACC__)
        self.rota_y = round(float(_items[6]), self.__FLOAT_ACC__)
        self.rota_z = round(float(_items[7]), self.__FLOAT_ACC__)
        self.rota_w = round(float(_items[8]), self.__FLOAT_ACC__)

        self.ts_ms = self.time // 1000000  # 毫秒级时间戳
        self.ts = self.ts_ms // 1000  # 秒级时间戳

    @property
    def loc_str(self):
        return '%.4f,%.4f' % (self.trans_x, self.trans_y)

    @property
    def out_str(self):
        return '%d,%.4f,%.4f' % (self.ts_ms, self.trans_x, self.trans_y)

    @property
    def item_info(self):
        return 'Time,Header,translation.x,translation.y,translation.z,rotation.x,rotation.y,rotation.z,rotation.w'


class OxIODIMU(object):
    """
    牛津大学PDR数据集OxIOD;
    iPhone手机采集的惯性数据;
    参阅http://deepio.cs.ox.ac.uk/
    """
    def __init__(self, line):
        self.__FLOAT_ACC__ = 4  # 浮点数精度
        self.__GRAVITY__ = 9.80665  # 地球表面标准重力
        _items = line.strip().split(',')
        self.time = round(float(_items[0]), 2)
        self.attitude_roll = round(float(_items[1]), self.__FLOAT_ACC__)
        self.attitude_pitch = round(float(_items[2]), self.__FLOAT_ACC__)
        self.attitude_yaw = round(float(_items[3]), self.__FLOAT_ACC__)
        self.rota_rate_x = round(float(_items[4]), self.__FLOAT_ACC__)  # 单位: 弧度每秒
        self.rota_rate_y = round(float(_items[5]), self.__FLOAT_ACC__)
        self.rota_rate_z = round(float(_items[6]), self.__FLOAT_ACC__)
        self.raw_gvt_x = round(float(_items[7]), self.__FLOAT_ACC__)  # 以G为计量单位
        self.raw_gvt_y = round(float(_items[8]), self.__FLOAT_ACC__)
        self.raw_gvt_z = round(float(_items[9]), self.__FLOAT_ACC__)
        self.raw_acc_x = round(float(_items[10]), self.__FLOAT_ACC__)  # 以G为计量单位
        self.raw_acc_y = round(float(_items[11]), self.__FLOAT_ACC__)
        self.raw_acc_z = round(float(_items[12]), self.__FLOAT_ACC__)
        self.mag_x = round(float(_items[13]), self.__FLOAT_ACC__)
        self.mag_y = round(float(_items[14]), self.__FLOAT_ACC__)
        self.mag_z = round(float(_items[15]), self.__FLOAT_ACC__)

        self.ts = int(self.time)
        self.ts_ms = int(self.time * 1000)

    @property
    def gyrx(self):
        return round(self.rota_rate_x * 180.0 / math.pi, self.__FLOAT_ACC__)

    @property
    def gyry(self):
        return round(self.rota_rate_y * 180.0 / math.pi, self.__FLOAT_ACC__)

    @property
    def gyrz(self):
        return round(self.rota_rate_z * 180.0 / math.pi, self.__FLOAT_ACC__)

    @property
    def accx(self):
        return round(self.raw_acc_x * self.__GRAVITY__ * -1.0 + self.gvtx, self.__FLOAT_ACC__)

    @property
    def accy(self):
        return round(self.raw_acc_y * self.__GRAVITY__ * -1.0 + self.gvty, self.__FLOAT_ACC__)

    @property
    def accz(self):
        return round(self.raw_acc_z * self.__GRAVITY__ * -1.0 + self.gvtz, self.__FLOAT_ACC__)

    @property
    def gvtx(self):
        return round(self.raw_gvt_x * self.__GRAVITY__ * -1.0, self.__FLOAT_ACC__)

    @property
    def gvty(self):
        return round(self.raw_gvt_y * self.__GRAVITY__ * -1.0, self.__FLOAT_ACC__)

    @property
    def gvtz(self):
        return round(self.raw_gvt_z * self.__GRAVITY__ * -1.0, self.__FLOAT_ACC__)

    @property
    def out_str(self):
        return '%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f' \
               % (self.ts_ms, self.accx, self.accy, self.accz, self.gyrx, self.gyry, self.gyrz, self.gvtx, self.gvty, self.gvtz)

    @property
    def item_info(self):
        return 'Time,attitude_roll(radians),attitude_pitch(radians),attitude_yaw(radians),' \
               'rotation_rate_x(radians/s),rotation_rate_y(radians/s),rotation_rate_z(radians/s),' \
               'gravity_x(G),gravity_y(G),gravity_z(G),user_acc_x(G),user_acc_y(G),user_acc_z(G),' \
               'magnetic_field_x(microteslas),magnetic_field_y(microteslas),magnetic_field_z(microteslas)'
