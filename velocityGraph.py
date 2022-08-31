import coord_utils
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

g = coord_utils.geoUtil()
distance = g.distance(40.226937,116.178958,40.227027,116.178973)

x = []
y = []

# for line in open('IPIN2022_T7_TestingTrial01.txt', "r"):
#     if line[:4] == 'GNSS':
#         gnssList = line.split(';')
#         time = eval(gnssList[1])
#         velocity = eval(gnssList[7])
#         x.append(time)
#         y.append(velocity)
# x = np.array(x)
# y = np.array(y)
f = open('speed.txt', 'r')
text1 = f.readline()[1:-2].split(',')
text1 = [float(x) for x in text1]

text2 = f.readline()[1:-2].split(',')
text2 = [float(x) for x in text2]
f.close()

l = len(text1)
x = np.arange(l)
y = np.array(text1)


plt.figure(figsize=(20, 10), dpi=400)
plt.plot(x, y, linewidth=1)
x = []
y = []

begin = False
pre_time = 0
pre_latitude = 0
pre_longitude = 0

# for line in open('IPIN2022_T7_TestingTrial01.txt', "r"):
#     if line[:4] == 'GNSS':
#         if not begin:
#             begin = True
#             gnssList = line.split(';')
#             pre_time = eval(gnssList[1])
#             pre_latitude = eval(gnssList[2])
#             pre_longitude = eval(gnssList[3])
#         else:
#             gnssList = line.split(';')
#             time = eval(gnssList[1])
#             latitude = eval(gnssList[2])
#             longitude = eval(gnssList[3])
#
#             distance = g.distance(pre_longitude, pre_latitude, longitude, latitude)
#             velocity = distance/(time-pre_time)
#
#             pre_time = time
#             pre_latitude = latitude
#             pre_longitude = longitude
#
#             x.append(time)
#             y.append(velocity)
# x = np.array(x)
# y = np.array(y)

l = len(text2)
x = np.arange(l)
y = np.array(text2)

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False

plt.xlabel('timestamp')
plt.ylabel('velocity')
plt.plot(x, y, linewidth=1, c='r')

blue_patch = mpatches.Patch(color='blue', label='预测')
red_patch = mpatches.Patch(color='red', label='真实')

plt.legend(handles=[blue_patch, red_patch])

plt.savefig('velocity.jpg')
plt.show()
