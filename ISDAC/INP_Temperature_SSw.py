### ISDAC 观测: INP浓度与温度关系图
### 读取多个 TXT 文件, 绘制散点图, 散点的颜色表示过饱和度(SSw)
### 时间: 2025-10-11
### 作者：付弘宇

import pathlib
import pandas as pd
import matplotlib.pyplot as plt

# 创建图形
fig, ax = plt.subplots(dpi=500)
ax.set_yscale('log')
ax.set_xlabel('Temperature(℃)')
ax.set_ylabel('INP concentration(cm-3)')# 不知道是不是标况下的INP浓度
ax.tick_params(which='both', top=True, bottom=True, left=True, right=True, direction='in')
ax.set_title('INP concentration vs Temperature')

data_dir = pathlib.Path("./data/ISDAC/brooks-cfdc")# 数据文件夹路径
csv_files = list(data_dir.glob('*.txt'))# 获取所有 TXT 文件
all_df = pd.DataFrame() # 用于存储所有数据

for csv_file in csv_files:
    df = pd.read_csv(csv_file, sep='\t').astype({'IN Conc (cm-3)': 'float', 'Aerosol T (C)': 'float'})

    df = df.dropna(subset=['IN Conc (cm-3)', 'Aerosol T (C)'])  # 删除IN浓度或温度为NaN的行

    all_df = pd.concat([all_df, df], ignore_index=True, verify_integrity=True)

#all_df = all_df.sort_values(by='Aerosol T (C)') # 按温度排序(加不加这一行, 对可视化没有影响)
x = all_df['Aerosol T (C)']
y = all_df['IN Conc (cm-3)']
c = all_df['SS.Water (%)']
sc = ax.scatter(x=x, y=y, c=c, cmap='Blues')

# 添加图例
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('SS.Water (%)')

plt.savefig("./master0_2025/ISDAC/INP_Temperature_SSw.png", bbox_inches='tight')