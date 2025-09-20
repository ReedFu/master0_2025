### AC3 观测: INP浓度与温度关系图
### 站点: Ny-Ålesund, Svalbard
### 读取一个 tab 数据文件, 绘制散点图
### 时间: 2025-09-20
### 作者：付弘宇

import pandas as pd
import matplotlib.pyplot as plt

# 创建图形
fig, ax = plt.subplots(dpi=500)
ax.set_yscale('log')
ax.set_xlabel('Temperature(℃)')
ax.set_ylabel('INP concentration(#/L)')
ax.tick_params(which='both', top=True, bottom=True, left=True, right=True, direction='in')
ax.set_title('INP concentration vs Temperature')

df = pd.read_table(r"D:\Data\PAMARCMiP\Wex-etal_2019_INP\datasets\NYA_INP.tab", skiprows=21)

x = df['T tech [°C]']
y = df['N INP air [#/l]']
ax.set_ylim(y[y>0].min()/2, y.max()*2)# 设置y轴范围
ax.scatter(x, y)

#ax.legend(fontsize='xx-small', loc='best')
plt.savefig("./master0_2025/PAMARCMiP/INP_vs_Temperature_NYA.png", bbox_inches='tight')