import math
import matplotlib.pyplot as plt
import numpy as np

# 求积分
# 参数：此处为陀螺仪y轴数据，积分周期（250Hz的倍数，因为题目为250Hz）
# 返回：各周期积分完的值
def intgr_gyr_cycle(gyr_axis_list, inte_cycle):
    """陀螺仪积分"""
    gyr_bear_diff_list = []
    for i in range(0, len(gyr_axis_list), inte_cycle):
        sec_gyr_axis_list = gyr_axis_list[i:i+inte_cycle]
        bear_diff = sum([x * 0.004 for x in sec_gyr_axis_list]) * 57.2957   # 角度和弧度换算
        gyr_bear_diff_list.append(bear_diff * -1.0)
    return gyr_bear_diff_list


# 读取数据（主要使用陀螺仪y轴数据）
# 参数：文件名，开始时间(秒)
# 返回：结果集合包括初始值，陀螺仪y轴数值，真值
def get_gyro_data(file_name, second):
    result = {}     # 返回结果集合

    flag = 1    # 标志位，用于控制读取初始值
    init_bear = 0.0     # 初始bearing
    gyro_axis_y = []    # GYRO中Y轴的数据
    gnss_bear_truth = []    # 真值

    times=0     

    with open(file_name, 'r') as f:
        line =f.readline()
        while line:
            line=f.readline()
            # 记录GNSS数据
            if line.startswith("GNSS") and flag==1:    
                if times >= second:     # 从第second秒开始，该秒后第一个GNSS的bearing作为初始值
                    init_bear = (float)(line.split(";")[5])
                    flag = 0       # 记录完初始位置标志置0
                else:
                    times+=1        # 不足second秒则跳过
            elif line.startswith("GNSS") and flag==0:    # 记录每次GNSS的bearing作为真值
                gnss_bear_truth.append((float)(line.split(";")[5]))
            # 记录陀螺仪y轴角速度
            if line.startswith("GYRO") and flag==0:      
                gyro_axis_y.append((float)(line.split(";")[-3]))
    # 将结果保存到集合
    result['init_bear'] = init_bear
    result['gyro_axis_y'] = gyro_axis_y
    result['gnss_bear_truth'] = gnss_bear_truth
    return result

# 直方图
# 参数：需要画图的值
def Histogram(list):
    plt.title("compensate value") 
    plt.xlabel("time(30min/pers)") 
    plt.ylabel("bearing(°)")  
    plt.hist(list,bins=10,edgecolor="blue",histtype="bar",alpha=0.5)
    plt.show()

# 测试
def main():
    res = get_gyro_data('trials/testing trial_1/IPIN2022_T7_TestingTrial01.txt', 300)   # 从数据集中得到初始角度，GYRO中的y轴的角速度，以及GNSS中的bearing
    init_bear = res['init_bear']        # 初始角度
    gyro_axis_y = res['gyro_axis_y']    # GYRO中的y轴的角速度
    gnss_bear_truth = res['gnss_bear_truth']    # 真值
    diff = []
    x_label = []

    # 根据角速度求积分（周期）
    cycle = 30      # 30秒一个周期
    inte_res_list = intgr_gyr_cycle(gyro_axis_y, cycle)

    for i in range(0, len(inte_res_list)):
        temp_truth = gnss_bear_truth[i*cycle]
        temp_pre = init_bear + inte_res_list[i]
        diff_v = abs(temp_pre - temp_truth)
        diff.append(diff_v)     # 保存差值，画图用
        x_label.append(i)

    Histogram(diff)
    
    pass

if __name__ == '__main__':
    main()
