# 绘制不同季节的气溶胶粒子谱分布

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= 1. 论文图表全局格式设置 =================

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] 
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman'

# ================= 2. 数据处理 =================
# 提取粒径区间列名，并将其转换为浮点数，作为 X 轴坐标
final_psd_df = pd.read_csv(r"D:\Coding\Data\Lanzhou_aerosol\SMPS+APS\final_psd(MAX_APS_DP=2w).csv", index_col=0, parse_dates=True)
status_df = pd.read_csv(r"D:\Coding\Data\Lanzhou_aerosol\SMPS+APS\instrument_status(MAX_APS_DP=2w).csv", index_col=0, parse_dates=True)
inp_df = pd.read_csv(r"D:\Coding\Data\Lanzhou_cfdc\processed\N_INP(202409-202509)v2.4.1.csv", index_col=0, parse_dates=True)

df= pd.merge_asof(inp_df, final_psd_df, left_index=True, right_index=True, direction='nearest', tolerance=pd.Timedelta('1h'))

size_columns = df.columns[9:]
diameters = np.array([float(col) for col in size_columns])

# 计算均值
seasonal_mean = df.groupby('season')[size_columns].mean()
# 计算标准差
seasonal_std = df.groupby('season')[size_columns].std()

season_styles = {
    'Spring': {'color': '#2ca02c', 'label': 'Spring', 'linestyle': '-'},
    'Summer': {'color': '#d62728', 'label': 'Summer', 'linestyle': '-'},
    'Autumn': {'color': '#ff7f0e', 'label': 'Autumn', 'linestyle': '-'},
    'Winter': {'color': '#1f77b4', 'label': 'Winter', 'linestyle': '-'},
}

# ================= 3. 开始绘图 =================
fig, ax = plt.subplots(figsize=(8, 6), dpi=500)

# 遍历每个季节进行绘制
for season in ['Spring', 'Summer', 'Autumn', 'Winter']:
    # 获取当前季节的均值和标准差
    if season not in seasonal_mean.index:
        continue
        
    y_values = seasonal_mean.loc[season].values
    y_std = seasonal_std.loc[season].values
    
    style = season_styles.get(season, {'color': 'black', 'label': season, 'linestyle': '-'})
    
    # 1. 绘制均值主线
    ax.plot(diameters, y_values, 
            color=style['color'], 
            linestyle=style['linestyle'],
            linewidth=2.0, 
            label=style['label'],
            zorder=3) # 确保实线在阴影上方
            
    # 2. 计算误差条的上下界
    upper_bound = y_values + y_std
    
    # 关键逻辑：如果 mean - std < 0，则下界取 mean (即不显示下半部分阴影)
    # 否则取 mean - std
    lower_bound = np.where((y_values - y_std) > 0, y_values - y_std, y_values)
    
    # 3. 绘制误差阴影
    ax.fill_between(diameters, lower_bound, upper_bound, 
                    color=style['color'], 
                    alpha=0.2,          # 透明度
                    edgecolor='none',   # 不显示阴影边缘线条
                    zorder=2)

# ================= 4. 图表美化与细节调整 =================
# 保持你之前的设置
ax.set_xscale('log')
ax.set_yscale('log')

# 设置坐标轴标签
ax.set_xlabel(r'$Particle \ Diameter \ D_p \ (nm)$', fontsize=14, fontweight='bold')
ax.set_ylabel(r'$dN/d\log D_p \ (\#/cm^3)$', fontsize=14, fontweight='bold')

# 刻度设置
ax.tick_params(axis='both', which='major', labelsize=12, length=6, direction='in')
ax.tick_params(axis='both', which='minor', length=3, direction='in')

ticks = [10, 100, 500, 1000, 2500, 10000]
ax.set_xlim(10, 20000)
ax.set_xticks(ticks)
ax.set_xticklabels([str(x) for x in ticks])

# 手动添加网格线
for x in [100, 500, 1000, 2500, 10000]:
    ax.axvline(x=x, color='gray', linestyle='--', alpha=0.3, zorder=1)

# 图例
ax.legend(loc='lower left', fontsize=12, frameon=False)

plt.tight_layout()
plt.savefig(r'D:\Coding\master0_2025\Thesis\fig 3.png', dpi=500,bbox_inches='tight')
print("图表已保存至: D:\\Coding\\master0_2025\\Thesis\\fig 3.png")