### PAMARCMiP 观测: INP浓度与温度关系图(未扣除背景值)
### 仪器: INDA(Ice Nucleation Droplet Array)
### 读取多个 tab 数据文件, 绘制散点图
### 时间: 2025-09-20
### 作者：付弘宇

import matplotlib.pyplot as plt
import pandas as pd
import pathlib

data_dir = pathlib.Path(r"D:\Data\PAMARCMiP\Hartmann-etal_2019_INP-PAMARCMIP\datasets")# 数据文件夹路径
data_files = list(data_dir.glob('*.tab'))

miny, maxy = 1, 0

fig = plt.figure(dpi=500)
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Temperature(℃)')
ax.set_ylabel('INP concentration(#/L)')
ax.tick_params(which='both', top=True, bottom=True, left=True, right=True, direction='in')
ax.set_title('INP concentration vs Temperature\nINDA(Ice Nucleation Droplet Array)')

for data_file in data_files:

    df = pd.read_table(data_file, skiprows=19)
    x = df[df.columns[0]]# Temperature (℃)
    y = df[df.columns[1]]# INP concentration (#/L)

    if min(y[y>0]) < miny:
        miny = min(y[y>0])
    if max(y) > maxy:
        maxy = max(y)
    ax.set_ylim(miny/2, maxy*2)

    ax.scatter(x,y,label=data_file.stem)

ax.legend(fontsize='xx-small', loc='best')
plt.savefig("./master0_2025/PAMARCMiP/INP_vs_Temperature_Hartmann_INDA.png", bbox_inches='tight')