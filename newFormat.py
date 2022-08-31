import random

import torch
import numpy as np

def get_data():
    begin = False
    curr_v = 0
    next_v = 0
    acceData = []
    acceDataList = []

    for line in open('IPIN2022_T7_TestingTrial01.txt', "r"):
        if line[:4] == 'GNSS':
            if not begin:
                begin = True
                curr_v = eval(line.split(';')[7])
            else:
                if len(acceData) < 1000:
                    next_v = eval(line.split(';')[7])

                    lable = int(round(next_v - curr_v, 1) * 10)
                    acceArray = list(map(float, acceData))
                    if len(acceArray) < 784:
                        acceArray = list(acceArray + [0.0] * (784 - len(acceArray)))
                    else:
                        acceArray = acceArray[:784]

                    acceArray = np.array(acceArray)
                    acceArray = acceArray.reshape(28, 28)
                    acceData = acceArray.tolist()
                    f = torch.tensor([acceData])
                    tup = (f, lable)
                    acceDataList.append(tup)

                    acceData = []
                    curr_v = next_v
                    next_v = 0
                else:
                    acceData = []
                    curr_v = eval(line.split(';')[7])
                    next_v = 0
        elif begin and line[:4] == 'ACCE':
            acceData.extend(line.split(';')[3:6])

    Dataset = acceDataList[0:2000]
    testset = acceDataList[2000:2300]
    return Dataset, testset

def reformatDataset(datasets, list):
    newDataset = []
    for dataset in datasets:
        if list.index(dataset[1]):
            dataset = (dataset[0], list.index(dataset[1]))
            newDataset.append(dataset)
        else:
            dataset = (dataset[0], list.index(0))
            newDataset.append(dataset)
    return newDataset



