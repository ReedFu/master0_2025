### AC3 观测: INP浓度与温度关系图
### 读取一个 tab 数据文件, 绘制散点图, 其中加热样品用红色(橙色)表示, 未加热样品用蓝色表示
### 未完成!!!
### 时间: 
### 作者: 

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
print(df.iloc[2, :])# 查看某一行数据
# 绘制散点图
for row in np.arange(rows):
    x = T
    y = df.iloc[row, 8:][::-1]  # 取第9列到最后一列, 并反转顺序
    #print(df['T:Temp descr (Thermal treatment (85°C) yes/no)'][row] == 'yes')
    #print(df['T:Temp descr (Thermal treatment (90°C) yes/no)'][row] == 'yes')

    if df['T:Temp descr (Thermal treatment (85°C) yes/no)'][row] == 'yes' :# 85℃加热过的样品
        ax.scatter(x, y, color='orange')
    elif df['T:Temp descr (Thermal treatment (90°C) yes/no)'][row] == 'yes' :# 90℃加热过的样品
        ax.scatter(x, y, color='red')
    elif df['T:Temp descr (Thermal treatment (85°C) yes/no)'][row] == 'no' and df['T:Temp descr (90°C) yes/no)'][row] == 'no' :# 未加热过的样品
        ax.scatter(x, y, color='blue')

#ax.legend(fontsize='xx-small', loc='best')
plt.savefig("./master0_2025/PAMARCMiP/INP_vs_Temperature_Sze_heated.png", bbox_inches='tight')