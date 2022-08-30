# function：给定漂移补偿时间，以第一次所得差值作为漂移量进行补偿
# 注：只有GNSS数据不间断情况下才能正确补偿
# start_time、end_time 为所取的GNSS数据序号

import matplotlib.pyplot as plt

trialsdir = "trials/TestingTrial01.txt"
start_time = 180
end_time = 900
sample_time = 0.004   # 采样频率250Hz
bearing = 0.0    # 航向
interval = 240   # 4 min 补偿一次漂移

if __name__ == '__main__':
    loop = 0
    drift = 0  # 累积漂移
    drift_check = False   # 漂移计算判断
    time = 0
    bearing_change = 0.0
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
        elif "GYRO" in data_list[i]:
            n = data_list[i].split(";")
            data_list[i] = []
            data_list[i].append(n[0])
            data_list[i].append(float(n[4]))

    i = 0
    fig = plt.figure()  # 创建画布
    ax = fig.add_subplot(111)
    while time < end_time+1:
        if data_list[i][0] == "GNSS":
            time += 1
            if time == start_time+1:
                bearing = data_list[i][1]
                p1 = ax.scatter(time - 1, data_list[i][1], marker='.', color='red', s=20, label='ground truth')
                p2 = ax.scatter(time - 1, bearing, marker='.', color='green', s=20, label='prediction')
            elif time > start_time+1:
                loop = (loop + 1) % interval
                bearing_change = 57.296 * bearing_change
                bearing = (bearing - bearing_change) % 360
                if loop == 0:
                    if drift_check is False:
                        drift = bearing - data_list[i][1]
                        drift_check = True
                        print('The first 4min drift is: ', drift)
                    bearing -= drift
                p1 = ax.scatter(time-1, data_list[i][1], marker='.', color='red', s=20)
                p2 = ax.scatter(time-1, bearing, marker='.', color='green', s=20)
                bearing_change = 0.0
        else:
            if time < start_time+1:
                i += 1
                continue
            bearing_change += data_list[i][1] * sample_time
        i += 1

    plt.xticks(range(start_time, end_time+1, (end_time-start_time)//10))
    plt.yticks(range(0, 360, 60))
    plt.xlabel("time")
    plt.ylabel("bearing")
    plt.legend(loc='best')
    plt.show()
    f.close()
