### ICE-D 观测实验: INP 浓度与温度的关系
### 读取多个 CSV 文件, 绘制散点图, 每个文件用不同的颜色来表示
### 时间：2025-08-27
### 作者：付弘宇

import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import matplotlib.colors as mcolors

# 创建图形
fig, ax = plt.subplots(dpi=500)
ax.set_yscale('log')
ax.set_xlabel('Temperature(℃)')
ax.set_ylabel('INP concentration(m^-3)')
ax.tick_params(which='both', top=True, bottom=True, left=True, right=True, direction='in')
ax.set_title('INP concentration vs Temperature')
colors = list(mcolors.XKCD_COLORS.values())

# 文件夹路径
data_dir = pathlib.Path(r"E:\data\ICE-D\INP_data")
csv_files = list(data_dir.glob('*.csv'))

for i, csv_file in enumerate(csv_files):
    df = pd.read_csv(csv_file, header=6)
    
    # 转换为数值类型，将错误值设为NaN
    df['T'] = pd.to_numeric(df['T'], errors='coerce')
    df['inp_air'] = pd.to_numeric(df['inp_air'], errors='coerce')
    
    # 移除包含NaN的行
    df_clean = df.dropna(subset=['T', 'inp_air'])
    
    ax.scatter(x=df_clean['T'], y=df_clean['inp_air'], label=csv_file.stem, color=colors[i % len(colors)])

# 添加图例
ax.legend(fontsize='xx-small', loc='best', ncol=5)

plt.savefig("./master0_2025/ICE-D/INP_vs_Temperature.png", bbox_inches='tight')
