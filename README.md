# IPIN 2022, Track 6

## 本地测试流程

运行脚本`offline_evaluation.ipynb`。 共有两种请求方式：
- 1、请求官方服务器，trialname为`T62288821d9d01`与`T62288821d9d02`；
- 2、使用`evaalapi.py`搭建本地服务器，编写yaml配置文件，请求本地服务器；

### 请求官方服务器

- 按`offline_evaluation.ipynb`流程走通即可；

### 请求本地服务器

- 1、项目目录下创建子目录`./trials/`， 将[网盘](https://mail.bjtu.edu.cn/coremail/XT5/jsp/download.jsp?share_link=0C5FF39956D9404A86F745CC5C752ECF&uid=19126355%40bjtu.edu.cn) （密码h3dd）提供的**txt**与**yaml**文件拷贝至子目录下；
- 2、运行脚本`python -u evaalapi.py`，默认启动的IP与端口号为`http://127.0.0.1:5000/`；
- 3、按`offline_evaluation.ipynb`流程走通即可；
