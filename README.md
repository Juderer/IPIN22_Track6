# IPIN 2022, Track 6
velocityGraph.py 读取 speed.txt的数据绘制出图像，保存为velocity.jpg

speed.txt由两个list构成 第一个是预测list，第二个是真实list

formatDataset.py 处理IPIN2022_T7_TestingTrial01.txt数据，生成格式化txt文件（速度由数据集给出）

newFormatDataset.py 处理IPIN2022_T7_TestingTrial01.txt数据，生成格式化txt文件（速度由经纬度计算）

newFormat.py 处理IPIN2022_T7_TestingTrial01.txt数据，生成格式化数据list，适配lstmTest.py

lstmTest.py 其他lstm模型 采用分类预测 效果较差

coord_utils.py 经纬度计算相关
