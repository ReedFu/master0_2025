### FIRE-ACE 观测: INP浓度与温度关系图
### 读取多个 TXT 文件, 绘制散点图, 每个文件用不同的颜色来表示
### 时间: 2025-09-06
### 作者：付弘宇

import pathlib
import pandas as pd
import matplotlib.pyplot as plt

# 创建图形
fig, ax = plt.subplots(dpi=500)
ax.set_yscale('log')
ax.set_xlabel('Temperature(℃)')
ax.set_ylabel('INP concentration(L^-3 at STP)')# 标况下的INP浓度
ax.tick_params(which='both', top=True, bottom=True, left=True, right=True, direction='in')
ax.set_title('INP concentration vs Temperature')
#ax.set_ylim(10**(-2),10**6)# 设置y轴范围

data_dir = pathlib.Path(r"E:\data\FIRE-ACE\CFDC_NCAR C130 Aircraft")# 数据文件夹路径
csv_files = list(data_dir.glob('fire*.txt'))# 获取所有 TXT 文件

for csv_file in csv_files:
    df = pd.read_csv(csv_file, skiprows=19, sep='\s+')
    
    # 去除IN浓度等于0.0的行
    df = df[df['INconc'] != 0.0]

    # 绘制散点图
    x = df['Ts']
    y = df['INconc']
    ax.scatter(x, y, label=csv_file.stem)

# 添加图例
ax.legend(fontsize='xx-small', loc='best')

#plt.show()
plt.savefig("./master0_2025/FIRE-ACE/INP_vs_Temperature.png", bbox_inches='tight')