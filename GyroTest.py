import matplotlib.pyplot as plt
import math

SensorTimestamp = 0.0
angle = [0.0,0.0,0.0]
init_bearing = 0.0
index = 0
first_init_flag = True
x = []
y = []
z = []
s = 0
sum_perid = 840
decomp_cycle = 120

# 处理度数，获得[-180, 180]内的结果    
def func(a):
    c = abs(a)%360
    if a < 0:
        b = (360-c) if (c > 180) else -c
    else:
        b = -(360-c) if (c > 180) else c
    return b
    
for line in open(r"IPIN2022_T7_TestingTrial01_GYRO.txt", "r") : # 设置文件对象并读取每一行文件
    if index/250.0 == sum_perid:
        break
    line_data = line.split(";")
    if line_data[0] == 'GNSS' and first_init_flag:
        init_bearing = float(line_data[5])
        first_init_flag = False
    if line_data[0] == 'GNSS':
        x.append(index/250.0)
        y.append(func(func(float(line_data[5]) - init_bearing)
                      - func(angle[1]*57.3)
                     )
                 # 补偿
#                  + 0.08*((index/250.0)%250.0)
                )
        z.append(func(angle[1]*57.3))
        if s == 120:
            angle = [0.0,0.0,0.0]
            init_bearing = float(line_data[5])
            s = 0
        s += 1
    
    if line_data[0] == 'GYRO':
        if SensorTimestamp != 0.0:
            dT = float(line_data[2]) - SensorTimestamp
            angle[1] -= float(line_data[4]) * dT
        SensorTimestamp = float(line_data[2])
        index += 1
        
#     print('Deviation: ',b)
    
# 绘图
plt.figure(figsize=(16, 12))   

# 按固定周期重组
new_y = []
new_x = []

print("x.len: ",len(x))
print("y.len: ",len(y))
cal_sum = 0.0

sum_of_cycle = sum_perid / decomp_cycle
print(type(math.floor(sum_of_cycle)))
for i in range(math.floor(sum_of_cycle)):
    for j in range(math.floor(decomp_cycle)):
        cal_sum += y[j+i*decomp_cycle]
    new_y.append(cal_sum/float(decomp_cycle))
    new_x.append((i+1)*decomp_cycle)
    cal_sum = 0.0

plt.plot(new_x,  # 横坐标
            new_y,  # 纵坐标
            c='red',  # 点的颜色
            linewidth=1)  
plt.scatter(new_x,  # 横坐标
            new_y,  # 纵坐标
            c='red',  # 点的颜色
            label='deviation')  

plt.xlabel('time/s')
 
plt.ylabel('deviation/°')
 
# 设置数字标签
for a, b in zip(new_x, new_y):
    plt.text(a, b, round(b,2), ha='center', va='bottom', fontsize=10)

plt.axhline(0, 0, sum_perid,color="blue")#横线

plt.legend()  # 显示图例
plt.savefig("./120sAverage.png",dpi=200)
plt.show()  # 显示所绘图形