#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2021/08/21 19:10
@Author     : wandergis, zhushuli
@File       : coord_utils.py
@DevTool    : PyCharm
@Desc       : 坐标系相关工具
@Refer      : https://github.com/wandergis/coordTransform_py
"""
import math
import torch
import numpy as np


class geoUtil():
    def __init__(self):
        self.x_pi = 3.14159265358979324 * 3000.0 / 180.0
        self.pi = 3.1415926535897932384626
        self.a = 6378245.0  # 长半轴
        self.ee = 0.00669342162296594323  # 偏心率平方
        self.Rc = 6378137.0
        self.Rj = 6356725.0

    # ----------------------- calculate bearing -----------------------
    def compute_wgs_dx_dy(self, lon1, lat1, lon2, lat2):
        DEG2RAD = self.pi / 180.0
        r_lat_1 = lat1 * DEG2RAD
        r_lng_1 = lon1 * DEG2RAD
        r_lat_2 = lat2 * DEG2RAD
        r_lng_2 = lon2 * DEG2RAD
        Ec = self.Rj + (self.Rc - self.Rj) * (90.0 - lat1) / 90.0
        Ed = Ec * math.cos(r_lat_1)
        dx = (r_lng_2 - r_lng_1) * Ed
        dy = (r_lat_2 - r_lat_1) * Ec
        return dx, dy

    def format_angle(self, rad_angle):
        scale = rad_angle / (2 * np.pi)
        if scale >= 1:
            rad_angle -= float(np.pi * 2 * int(scale))
        elif scale <= -1:
            rad_angle += float(np.pi * 2 * int(scale))
        if rad_angle < 0:
            return float(np.pi * 2) + rad_angle
        return rad_angle

    def compute_loc_angle(self, lon1, lat1, lon2, lat2):
        ignore_dist = 0.1
        dx, dy = self.compute_wgs_dx_dy(lon1, lat1, lon2, lat2)
        if np.fabs(dx) <= ignore_dist and np.fabs(dy) <= ignore_dist:
            return -1
        else:
            direction = self.format_angle(float(np.arctan2(dx, dy)))
            return direction * float(180.0 / np.pi)

    # ----------------------- coordinate system transformation functions -----------------------
    def gcj02_to_bd09(self, lng, lat):
        """
        火星坐标系(GCJ-02)转百度坐标系(BD-09)
        谷歌、高德——>百度
        :param lng:火星坐标经度
        :param lat:火星坐标纬度
        :return:
        """
        z = math.sqrt(lng * lng + lat * lat) + 0.00002 * math.sin(lat * self.x_pi)
        theta = math.atan2(lat, lng) + 0.000003 * math.cos(lng * self.x_pi)
        bd_lng = z * math.cos(theta) + 0.0065
        bd_lat = z * math.sin(theta) + 0.006
        return [bd_lng, bd_lat]

    def bd09_to_gcj02(self, bd_lon, bd_lat):
        """
        百度坐标系(BD-09)转火星坐标系(GCJ-02)
        百度——>谷歌、高德
        :param bd_lat:百度坐标纬度
        :param bd_lon:百度坐标经度
        :return:转换后的坐标列表形式
        """
        x = bd_lon - 0.0065
        y = bd_lat - 0.006
        z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * self.x_pi)
        theta = math.atan2(y, x) - 0.000003 * math.cos(x * self.x_pi)
        gg_lng = z * math.cos(theta)
        gg_lat = z * math.sin(theta)
        return [gg_lng, gg_lat]

    def wgs84_to_gcj02(self, lng, lat):
        """
        WGS84转GCJ02(火星坐标系)
        :param lng:WGS84坐标系的经度
        :param lat:WGS84坐标系的纬度
        :return:
        """
        if self.__out_of_china(lng, lat):  # 判断是否在国内
            return [lng, lat]
        dlat = self.__transformlat(lng - 105.0, lat - 35.0)
        dlng = self.__transformlng(lng - 105.0, lat - 35.0)
        radlat = lat / 180.0 * self.pi
        magic = math.sin(radlat)
        magic = 1 - self.ee * magic * magic
        sqrtmagic = math.sqrt(magic)
        dlat = (dlat * 180.0) / ((self.a * (1 - self.ee)) / (magic * sqrtmagic) * self.pi)
        dlng = (dlng * 180.0) / (self.a / sqrtmagic * math.cos(radlat) * self.pi)
        mglat = lat + dlat
        mglng = lng + dlng
        return [mglng, mglat]

    def gcj02_to_wgs84(self, lng, lat):
        """
        GCJ02(火星坐标系)转GPS84
        :param lng:火星坐标系的经度
        :param lat:火星坐标系纬度
        :return:
        """
        if self.__out_of_china(lng, lat):
            return [lng, lat]
        dlat = self.__transformlat(lng - 105.0, lat - 35.0)
        dlng = self.__transformlng(lng - 105.0, lat - 35.0)
        radlat = lat / 180.0 * self.pi
        magic = math.sin(radlat)
        magic = 1 - self.ee * magic * magic
        sqrtmagic = math.sqrt(magic)
        dlat = (dlat * 180.0) / ((self.a * (1 - self.ee)) / (magic * sqrtmagic) * self.pi)
        dlng = (dlng * 180.0) / (self.a / sqrtmagic * math.cos(radlat) * self.pi)
        mglat = lat + dlat
        mglng = lng + dlng
        return [lng * 2 - mglng, lat * 2 - mglat]

    def bd09_to_wgs84(self, bd_lon, bd_lat):
        lon, lat = self.bd09_to_gcj02(bd_lon, bd_lat)
        return self.gcj02_to_wgs84(lon, lat)

    def wgs84_to_bd09(self, lon, lat):
        lon, lat = self.wgs84_to_gcj02(lon, lat)
        return self.gcj02_to_bd09(lon, lat)

    def __transformlat(self, lng, lat):
        ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
              0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
        ret += (20.0 * math.sin(6.0 * lng * self.pi) + 20.0 *
                math.sin(2.0 * lng * self.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lat * self.pi) + 40.0 *
                math.sin(lat / 3.0 * self.pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(lat / 12.0 * self.pi) + 320 *
                math.sin(lat * self.pi / 30.0)) * 2.0 / 3.0
        return ret

    def __transformlng(self, lng, lat):
        ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
              0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
        ret += (20.0 * math.sin(6.0 * lng * self.pi) + 20.0 *
                math.sin(2.0 * lng * self.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lng * self.pi) + 40.0 *
                math.sin(lng / 3.0 * self.pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(lng / 12.0 * self.pi) + 300.0 *
                math.sin(lng / 30.0 * self.pi)) * 2.0 / 3.0
        return ret

    def __out_of_china(self, lng, lat):
        """
        判断是否在国内，不在国内不做偏移
        :param lng:
        :param lat:
        :return:
        """
        return not (lng > 73.66 and lng < 135.05 and lat > 3.86 and lat < 53.55)

    # ----------------------- position inference functions -----------------------
    def position_from_angle_and_dist(self, old_lon, old_lat, anlge, dist):
        DEG2RAD = self.pi / 180.0
        radian = DEG2RAD * anlge
        dx = dist * math.sin(radian)
        dy = dist * math.cos(radian)
        return self.__position_from_offset(old_lon, old_lat, dx, dy)

    def __position_from_offset(self, old_lon, old_lat, offsetx, offsety):
        if (math.fabs(offsetx) < 0.000000001 and math.fabs(offsety) < 0.000000001):
            return old_lon, old_lat
        lng1 = old_lon
        lat1 = old_lat

        Rc = 6378137.0;
        Rj = 6356725.0;
        DEG2RAD = self.pi / 180.0;
        r_lat_1 = lat1 * DEG2RAD;
        r_lng_1 = lng1 * DEG2RAD;

        Ec = Rj + (Rc - Rj) * (90.0 - lat1) / 90.0;
        Ed = Ec * math.cos(r_lat_1);

        r_lng_2 = offsetx / Ed + r_lng_1;
        r_lat_2 = offsety / Ec + r_lat_1;

        new_lon_lat_lon = r_lng_2 / DEG2RAD;
        new_lon_lat_lat = r_lat_2 / DEG2RAD;

        return new_lon_lat_lon, new_lon_lat_lat

    def distance(self, lon1, lat1, lon2, lat2):
        return self.haversine_formula(lon1, lat1, lon2, lat2)

    def haversine_formula(self, lon1, lat1, lon2, lat2):
        """
        利用haversine公式计算球面上两点的直线距离
        haversine公式参考wikipedia：https://en.wikipedia.org/wiki/Haversine_formula
        :param lon1: 第一个点的经度
        :param lat1: 第一个点的纬度
        :param lon2: 第二个点的经度
        :param lat2: 第二个点的纬度
        :return: 两点之间的直线距离（单位：m）
        """
        # 地球半径（单位：m）
        earth_radius = 6.371e6
        # 将各个十进制度数转化为弧度
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
        # haversine公式
        hav = pow(math.sin((lat2 - lat1) / 2), 2) + \
              math.cos(lat1) * math.cos(lat2) * pow(math.sin((lon2 - lon1) / 2), 2)
        temp = math.asin(math.sqrt(hav)) * 2
        distance = earth_radius * temp

        return round(distance, 6)

    def haversine_formula_tensor(self, lon1, lat1, lon2, lat2):
        """
        利用haversine公式计算球面上两点的直线距离
        haversine公式参考wikipedia：https://en.wikipedia.org/wiki/Haversine_formula
        :param lon1: 第一个点的经度
        :param lat1: 第一个点的纬度
        :param lon2: 第二个点的经度
        :param lat2: 第二个点的纬度
        :return: 两点之间的直线距离（单位：m）
        """
        # 地球半径（单位：m）
        earth_radius = 6.371e6
        # 将各个十进制度数转化为弧度
        lon1, lat1, lon2, lat2 = [x * self.pi / 180.0 for x in [lon1, lat1, lon2, lat2]]
        # haversine公式
        hav = torch.pow(torch.sin((lat2 - lat1) / 2), 2) + \
              torch.cos(lat1) * torch.cos(lat2) * torch.pow(torch.sin((lon2 - lon1) / 2), 2)
        temp = torch.asin(torch.sqrt(hav)) * 2
        distance = earth_radius * temp

        return distance


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class AreaUtil:
    def __init__(self):
        pass

    @staticmethod
    def is_on_segment(pt, s_pt, e_pt):
        """ 利用向量叉乘判断点是否在线段上 """
        eps = 1e-8
        k = (e_pt.x - s_pt.x) * (pt.y - s_pt.y) - (e_pt.y - s_pt.y) * (pt.x - s_pt.x)
        return abs(k) < eps and pt.x < max(s_pt.x, e_pt.x) + eps and pt.x > min(s_pt.x, e_pt.x) - eps \
               and pt.y < max(s_pt.y, e_pt.y) + eps and pt.y > min(s_pt.y, e_pt.y) - eps

    @staticmethod
    def is_point_in_polygon(pt, polygon):  # (lon,lat), ((lon1,lat1),(lon2,lat2),...,(lon1,lat1))
        # 初始化数据
        pt = Point(pt[0], pt[1])
        polygon_pts = [Point(u[0], u[1]) for u in polygon]
        # 射线法
        eps = 1e-8
        num = 0
        for i in range(len(polygon_pts) - 1):
            if AreaUtil.is_on_segment(pt, polygon_pts[i], polygon_pts[i + 1]):
                return True
            k = (polygon_pts[i + 1].x - polygon_pts[i].x) * (pt.y - polygon_pts[i].y) \
                - (polygon_pts[i + 1].y - polygon_pts[i].y) * (pt.x - polygon_pts[i].x)
            d1 = polygon_pts[i].y - pt.y
            d2 = polygon_pts[i + 1].y - pt.y
            if k > eps and d1 < eps and d2 > eps:
                num += 1
            if k < -eps and d2 < eps and d1 > eps:
                num -= 1
        return num != 0


geo_util = geoUtil()

__all__ = ['geo_util', 'AreaUtil']
