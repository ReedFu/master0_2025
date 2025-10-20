### ACAPEX 飞机观测: INP浓度与温度关系图
### 读取多个 ICT 文件, 绘制散点图. 
### 时间: 
### 作者：付弘宇

import pathlib
import icartt
import matplotlib.pyplot as plt
import numpy as np

# 创建图形
fig, ax = plt.subplots(dpi=500)
ax.set_yscale('log')
ax.set_xlabel('Temperature(℃)')
ax.set_ylabel('INP concentration(cm-3)')# 不知道是不是标况下的INP浓度
ax.tick_params(which='both', top=True, bottom=True, left=True, right=True, direction='in')
ax.set_title('INP concentration vs Temperature')

# 文件夹路径
data_dir = pathlib.Path('./data/ACAPEX/demott-cfdc')
# 获取所有.ict文件
ict_files = list(data_dir.glob('*.ict'))
