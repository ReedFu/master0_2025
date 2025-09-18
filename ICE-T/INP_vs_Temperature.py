### ICE-T 观测: INP浓度与温度关系图
### 读取多个 TXT 文件, 绘制散点图, 每个文件用不同的颜色来表示
### 时间: 2025-08-28
### 作者：付弘宇

import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 创建图形
fig, ax = plt.subplots(dpi=500)
ax.set_yscale('log')
ax.set_xlabel('Temperature(℃)')
ax.set_ylabel('INP concentration(L^-3)')
ax.tick_params(which='both', top=True, bottom=True, left=True, right=True, direction='in')
ax.set_title('INP concentration vs Temperature')
#ax.set_xlim(-30,-12)
#ax.set_ylim(1,10**6)
colors = list(mcolors.XKCD_COLORS.values())


data_dir = pathlib.Path(r"E:\data\ICE-T\INP")# 数据文件夹路径
csv_files = list(data_dir.glob('CSU*.txt'))# 获取所有以 CSU 开头的 TXT 数据文件
columns = ['STRT_DATE/TIME','STOP_DATE/TIME','LAT','LON','ALT','CVI','CVIF',
'AER','AER_S','IN_RAW','IN_COR','IN_COR_S','IN_SIG','T','T_S','W_SAT','W_SAT_S','P','P_S']# 手动设置列名



for i, csv_file in enumerate(csv_files):
    df = pd.read_csv(csv_file, names=columns, skiprows=23, sep='\s+')

    # 数据清洗
    df = df.replace(9999, np.nan)#将负值数据(文件中用9999表示)替换为NaN
    df = df.dropna(subset=['IN_COR', 'T'])

    # 绘制散点图
    x = df['T']
    y = df['IN_COR']

    ax.scatter(x,y,label=csv_file.stem, color=colors[i*4 % len(colors)])# 每隔4个颜色取一个，避免颜色过于相似

# 添加图例
ax.legend(fontsize='xx-small', loc='best')

#plt.show()
plt.savefig("./master0_2025/ICE-T/INP_vs_Temperature.png", bbox_inches='tight')