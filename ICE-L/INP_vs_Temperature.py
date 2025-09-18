### ICE-L 观测: INP浓度与温度关系图
### 读取多个 TXT 文件, 绘制散点图, 每个文件用不同的颜色来表示
### 时间: 2025-08-29
### 作者：付弘宇

import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 创建图形
fig, ax = plt.subplots(dpi=500)
ax.set_yscale('log')
ax.set_xlabel('Temperature(℃)')
ax.set_ylabel('INP concentration(cm^-3)')
ax.tick_params(which='both', top=True, bottom=True, left=True, right=True, direction='in')
ax.set_title('INP concentration vs Temperature')
#ax.set_xlim(-30,-12)
ax.set_ylim(10**(-4.5),10)


data_dir = pathlib.Path(r'E:\data\ICE-L\cfdc\dataGkHRuM')# 数据文件夹路径
csv_files = list(data_dir.glob('*.txt'))# 获取所有 TXT 文件

for csv_file in csv_files:
    df = pd.read_csv(csv_file, skiprows=33, sep='\s+')# 读取数据

    # 数据清洗
    df = df.replace(999.9999, np.nan) #将负值数据(文件中用999.9999表示)替换为NaN
    #df = df.replace(0.0000, np.nan) #将0值数据替换为NaN #加上这一行没反应??
    df = df.dropna(subset=['INConc', 'TCFDC'])
    #df['INConc'] = df['INConc'] * 1000 # 将 INP 浓度从转 cm^-3 转换为 L^-3 

    # 绘制散点图
    x = df['TCFDC']
    y = df['INConc']
    ax.scatter(x,y,label=csv_file.stem)

# 添加图例
ax.legend(fontsize='xx-small', loc='best')

#plt.show()
plt.savefig("./master0_2025/ICE-L/INP_vs_Temperature.png", bbox_inches='tight') 