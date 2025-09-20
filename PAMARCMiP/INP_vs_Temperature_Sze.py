### AC3 观测: INP浓度与温度关系图
### Sze 等人 2 年的离线观测数据
### 读取一个 tab 数据文件, 绘制散点图
### 时间: 2025-09-18
### 作者：付弘宇

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 创建图形
fig, ax = plt.subplots(dpi=500)
ax.set_yscale('log')
ax.set_xlabel('Temperature(℃)')
ax.set_ylabel('INP concentration(#/L)')
ax.tick_params(which='both', top=True, bottom=True, left=True, right=True, direction='in')
ax.set_title('INP concentration vs Temperature')
#ax.set_ylim(10**(-2),10**6)# 设置y轴范围

T = np.arange(-30.4, -4.4, 0.1)# 自行设置温度区间
df = pd.read_table(r"D:\Data\PAMARCMiP\Sze-etal_2023_NINP.tab", skiprows=286)
rows, cols = df.shape

# 绘制散点图
for row in np.arange(rows):
    x = T
    y = df.iloc[row, 8:][::-1]  # 取第9列到最后一列, 并反转顺序
    ax.scatter(x, y
               #,label=df.iloc[row, 0]  # 用第一列作为标签(Sample label), 但标签太多，故弃用
    )

#ax.legend(fontsize='xx-small', loc='best')
plt.savefig("./master0_2025/PAMARCMiP/INP_vs_Temperature_Sze.png", bbox_inches='tight')