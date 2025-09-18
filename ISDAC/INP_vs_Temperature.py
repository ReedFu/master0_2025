### ISDAC 观测: INP浓度与温度关系图
### 读取多个 TXT 文件, 绘制散点图, 每个文件用不同的颜色来表示
### 时间: 2025-09-06
### 作者：付弘宇

import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 创建图形
fig, ax = plt.subplots(dpi=500)
#ax.set_yscale('log')
ax.set_xlabel('Temperature(℃)')
ax.set_ylabel('INP concentration(cm-3)')# 不知道是不是标况下的INP浓度
ax.tick_params(which='both', top=True, bottom=True, left=True, right=True, direction='in')
ax.set_title('INP concentration vs Temperature')
#ax.set_ylim(10**(-4),10**(-1))# 经调试后确定y轴范围

data_dir = pathlib.Path(r"E:\data\ISDAC\brooks-cfdc")# 数据文件夹路径
csv_files = list(data_dir.glob('*.txt'))# 获取所有 TXT 文件
#colomns = ['time', 'CFDC Temperature', 'Sat. Water', 'Sat. Ice', 'CFDC Pressure', 'Filter', 'IN Concentration']  # 列名
colors = list(mcolors.XKCD_COLORS.values())

for i, csv_file in enumerate(csv_files):
    df = pd.read_csv(csv_file, sep='\t')
    #print(df.columns) # 调试需要
    #print(df.info())
    #print(df)

    # 数据清洗
    df = df.replace(['-NaN', 'NaN'], np.nan)  # 将 '-NaN' 和 'NaN' 字符串替换为实际的 NaN 值
    df = df.dropna(subset=['IN Conc (cm-3)', 'Aerosol T (C)'])  # 删除IN浓度或温度为NaN的行
    #df = df[df['IN Concentration'] != 0.0] # 调试需要

    # 绘制散点图
    x = df['Aerosol T (C)']
    y = df['IN Conc (cm-3)']
    print(min(y), max(y)) # 调试需要
    ax.scatter(x, y, label=csv_file.stem, color=colors[i*5 % len(colors)]) # 每隔5个颜色取一个，避免颜色过于相似

# 添加图例
ax.legend(fontsize='xx-small', loc='best', ncol=4)

#plt.show()
plt.savefig("./master0_2025/ISDAC/INP_vs_Temperature.png", bbox_inches='tight')