import coord_utils


g = coord_utils.geoUtil()

curr_bp = []
acceData = []
gyroData = []
acceDataLine = []
gyroDataLine = []

next_bp = []
DataList = []
for line in open('IPIN2022_T7_TestingTrial01.txt', "r"):
    if line[:4] == 'GNSS':
        if not curr_bp:
            temp = line.split(';')
            curr_bp = temp[2:6]
            curr_bp[2] = temp[7]
        else:
            temp = line.split(';')
            next_bp = temp[2:6]
            next_bp[2] = temp[7]

            accelength = len(acceData)
            if accelength < 246:
                tempAcceData = acceData[accelength - 1]
                for i in range(246-accelength):
                    acceData.append(tempAcceData)
            for i in range(50):
                acceDataLine.extend(acceData[i*5])
            # print(len(acceDataLine))

            gyrolength = len(gyroData)
            if gyrolength < 246:
                tempGyroData = gyroData[gyrolength - 1]
                for i in range(246-gyrolength):
                    gyroData.append(tempGyroData)
            for i in range(50):
                gyroDataLine.extend(gyroData[i*5])
            # print(len(gyroDataLine))

            glist = ['0']
            glist = glist*150

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
    elif curr_bp and line[:4] == 'ACCE':
        acceData.append(line.split(';')[3:6])

    elif curr_bp and line[:4] == 'GYRO':
        gyroData.append(line.split(';')[3:6])

f = open("lstm_train_dataset_flat.txt", "a")
f.seek(0)
f.truncate()
for i in range(2000):
    line = ','.join(DataList[i])
    f.write(line+'\n')
f.close()

f = open("lstm_eval_dataset_flat.txt", "a")
f.seek(0)
f.truncate()
for i in range(200):
    line = ','.join(DataList[i+2000])
    line = line[1:]
    line = str(i)+line+','+str(i+1)
    f.write(line+'\n')
f.close()