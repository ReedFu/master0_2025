### M-PACE 观测: INP浓度与温度关系图
### 读取多个 TXT 文件, 绘制散点图, 散点的颜色表示过饱和度(SSw)
### 时间: 2025-09-06
### 作者：付弘宇

import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 创建图形
fig, ax = plt.subplots(dpi=500)
ax.set_yscale('log')
ax.set_xlabel('Temperature(℃)')
ax.set_ylabel('INP concentration(cm-3, CFDC T & P)')# 标况下的INP浓度
ax.tick_params(which='both', top=True, bottom=True, left=True, right=True, direction='in')
ax.set_title('INP concentration vs Temperature')
ax.set_ylim(10**(-4),10**(-1))# 经调试后确定y轴范围

data_dir = pathlib.Path(r"E:\data\M-PACE\demott-cfdc")# 数据文件夹路径
csv_files = list(data_dir.glob('2004*.txt'))# 获取所有 TXT 文件
colomns = ['time', 'CFDC Temperature', 'Sat. Water', 'Sat. Ice', 'CFDC Pressure', 'Filter', 'IN Concentration']  # 列名
all_df = pd.DataFrame()

for csv_file in csv_files:
    df = pd.read_csv(csv_file, names=colomns , skiprows=25, sep='\s+')
    
    # 数据清洗
    df = df.replace(99999, np.nan)  # 将缺失值(文件中为99999)替换为NaN
    df = df.dropna(subset=['IN Concentration', 'CFDC Temperature'])  # 删除IN浓度或温度为NaN的行

    df = df[['CFDC Temperature', 'IN Concentration', 'Sat. Water']]  # 选择所需的列
    all_df = pd.concat([all_df, df], ignore_index=True, verify_integrity=True)

# 绘制散点图
x = all_df['CFDC Temperature']
y = all_df['IN Concentration']
c = all_df['Sat. Water']
#print(min(c), max(c))
scatter = ax.scatter(x=x, y=y, c=c, cmap="Blues")

# 添加颜色条
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('Sat. Water(%)')

#plt.show()
plt.savefig("./master0_2025/M-PACE/INP_Temperature_SSw.png", bbox_inches='tight')