import coord_utils


def get_data(filename):
    g = coord_utils.geoUtil()

    begin = False
    begin1 = False

    curr_bp = []
    pre_time = 0
    pre_ati = 0
    pre_lon = 0
    pre_v = 0

    pre1_time = 0
    pre1_ati = 0
    pre1_lon = 0
    pre1_v = 0

    acceData = []
    gyroData = []
    acceDataLine = []
    gyroDataLine = []
    next_bp = []
    DataList = []
    for line in open(filename, "r"):
        if line[:4] == 'GNSS':
            if not begin:
                begin = True
                temp = line.split(';')
                pre_time = eval(temp[1])
                pre_ati = eval(temp[2])
                pre_lon = eval(temp[3])
            elif not begin1:
                begin1 = True
                temp = line.split(';')
                pre1_time = eval(temp[1])
                pre1_ati = eval(temp[2])
                pre1_lon = eval(temp[3])
                curr_bp = temp[2:6]
                distance = g.distance(pre_lon, pre_ati, pre1_lon, pre1_ati)
                v = distance / (pre1_time - pre_time)
                curr_bp[2] = str(v)
                if pre1_time - pre_time > 1.5 or v > 33:
                    begin = False
                    begin1 = False
            else:
                temp = line.split(';')
                next_bp = temp[2:6]
                time = eval(temp[1])
                ati = eval(temp[2])
                lon = eval(temp[3])

                distance = g.distance(pre1_lon, pre1_ati, lon, ati)
                v = distance / (time - pre1_time)

                pre1_time = time
                pre1_ati = ati
                pre1_lon = lon

                if time - pre1_time > 1.5 or v > 33:
                    begin = False
                    begin1 = False
                else:
                    next_bp[2] = str(v)
                    accelength = len(acceData)
                    if accelength < 250:
                        tempAcceData = acceData[accelength - 1]
                        for i in range(250 - accelength):
                            acceData.append(tempAcceData)
                    for i in range(250):
                        acceDataLine.extend(acceData[i])
                    # print(len(acceDataLine))

                    gyrolength = len(gyroData)
                    if gyrolength < 250:
                        tempGyroData = gyroData[gyrolength - 1]
                        for i in range(250 - gyrolength):
                            gyroData.append(tempGyroData)
                    for i in range(250):
                        gyroDataLine.extend(gyroData[i])

                    aline = []
                    aline.extend('1')
                    aline.extend(curr_bp)
                    aline.extend(next_bp)
                    aline.extend(acceDataLine)
                    aline.extend(gyroDataLine)
                    DataList.append(aline)
                    curr_bp = next_bp

                acceData = []
                acceDataLine = []
                gyroData = []
                gyroDataLine = []
                next_bp = []

        elif begin and begin1 and line[:4] == 'ACCE':
            acceData.append(line.split(';')[3:6])

        elif begin and begin1 and line[:4] == 'GYRO':
            gyroData.append(line.split(';')[3:6])

    return DataList

DataList = []
DataList.extend(get_data('IPIN2022_T7_TestingTrial01.txt'))
DataList.extend(get_data('IPIN2022_T7_TestingTrial02.txt'))

print(len(DataList))
f = open("lstm_train_dataset_flat.txt", "a")
f.seek(0)
f.truncate()
for i in range(4400):
    line = ','.join(DataList[i])
    f.write(line + '\n')
f.close()

f = open("lstm_eval_dataset_flat.txt", "a")
f.seek(0)
f.truncate()
for i in range(500):
    line = ','.join(DataList[i + 4400])
    line = line[1:]
    line = str(i) + line + ',' + str(i + 1)
    f.write(line + '\n')
f.close()

f = open("test.txt", "a")
f.seek(0)
f.truncate()
for i in range(500):
    line = ','.join(DataList[i + 4900])
    line = line[1:]
    line = str(i) + line + ',' + str(i + 1)
    f.write(line + '\n')
f.close()
