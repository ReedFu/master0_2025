### FIRE-ACE 观测: INP浓度与温度关系图
### 读取多个 TXT 文件, 绘制散点图, 散点的颜色表示过饱和度(SSw)
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
ax.set_ylim(10**(-2),10**6)

data_dir = pathlib.Path(r"E:\data\FIRE-ACE\CFDC_NCAR C130 Aircraft")# 数据文件夹路径
csv_files = list(data_dir.glob('fire*.txt'))# 获取所有 TXT 文件
all_df = pd.DataFrame()

for csv_file in csv_files:
    df = pd.read_csv(csv_file, skiprows=19, sep='\s+')
    df = df[['Ts', 'INconc', 'SSw']]  # 选择所需的列
    all_df = pd.concat([all_df, df], ignore_index=True, verify_integrity=True)

# 绘制散点图
x = all_df['Ts']
y = all_df['INconc']
c = all_df['SSw']
#print(min(c), max(c))
scatter = ax.scatter(x=x, y=y, c=c, cmap="Blues")

# 添加颜色条
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('SSw(%)')

#plt.show()
plt.savefig("./master0_2025/FIRE-ACE/INP_Temperature_SSw.png", bbox_inches='tight')