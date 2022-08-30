# function: 给定间隔，计算gyroscope漂移量
# 注：只有在GNSS数据不间断情况下才能正确计算漂移  已验证trial01数据集 458230 - 459387s （大约19分钟）内GNSS数据连续
# start_time、end_time 为GNSS数据所对应的GPS_TOW

import matplotlib.pyplot as plt
import numpy as np

trialsdir = "trials/TestingTrial01.txt"
start_time = 458230
end_time = 459250
sample_time = 0.004   # 采样频率250Hz
bearing = 0.0    # 航向
interval = 120   # 漂移计算间隔 30s, 1min, 2min


if __name__ == '__main__':
    drift = []  # 记录累积漂移
    flag = False
    deltaBearing = 0.0     # 1s 内航向变化量
    bearing_change = []    # 保存 deltaBearing
    deltaBearing_total = 0.0   # 间隔内航向变化
    gnss_seq = []
    prior = 0
    data_list = []
    f = open(trialsdir, 'r')
    file_data = f.readlines()
    for item in file_data:
        data = item.strip('\n')
        if "GYRO" in item or "GNSS" in item:
            data_list.append(data)

    for index, item in enumerate(data_list):
        if "GNSS" in item:
            del data_list[0: index]
            break

    for i in range(len(data_list)):
        if "GNSS" in data_list[i]:
            n = data_list[i].split(";")
            data_list[i] = []
            data_list[i].append(n[0])
            data_list[i].append(float(n[5]))
            data_list[i].append(int(float(n[8])))
        elif "GYRO" in data_list[i]:
            n = data_list[i].split(";")
            data_list[i] = []
            data_list[i].append(n[0])
            data_list[i].append(float(n[4]))

    i = 0
    while 1:
        if data_list[i][0] == "GNSS":
            if data_list[i][2] == start_time:
                gnss_seq.append(i)
                flag = True
            if data_list[i][2] > start_time:
                if data_list[i][2] <= end_time:
                    gnss_seq.append(i)
                bearing_change.append(deltaBearing)
                deltaBearing = 0.0
                if data_list[i][2] >= start_time + interval:
                    # 输出预测值与真值之差
                    if data_list[i][2] == start_time + interval:
                        for item in range(interval):
                            deltaBearing_total += bearing_change[item]
                        prediction = data_list[gnss_seq[prior]][1] - 57.296 * deltaBearing_total
                        prediction = prediction % 360
                        diff = data_list[i][1] - prediction
                        if diff < -300:
                            diff = 360 + diff
                        if diff > 300:
                            diff = -(360 - diff)
                        drift.append(diff)
                        prior += 1
                    else:
                        deltaBearing_total = deltaBearing_total - bearing_change[prior-1] + bearing_change[prior+interval-1]
                        prediction = data_list[gnss_seq[prior]][1] - 57.296 * deltaBearing_total
                        prediction = prediction % 360
                        diff = data_list[i][1] - prediction
                        if diff < -300:
                            diff = 360 + diff
                        if diff > 300:
                            diff = -(360 - diff)
                        drift.append(diff)
                        prior += 1
                    if data_list[i][2] == end_time + interval:
                        break
        if flag is False:
            i += 1
            continue
        if data_list[i][0] == "GYRO":
            deltaBearing += data_list[i][1] * sample_time
        i += 1

    average = 0.0
    for item in drift:
        average += item
        print(item)
    average = average / len(drift)
    print("average drift is: ", average)

    # 画图
    plt.hist(drift, bins=150, edgecolor="b", histtype="bar", alpha=0.5)
    plt.title("interval: %ds, average: %f" % (interval, average))
    plt.xlabel("drift")
    plt.ylabel("number")
    plt.show()
    f.close()
