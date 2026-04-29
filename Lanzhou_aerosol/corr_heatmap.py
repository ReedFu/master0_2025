# Draw a heatmap of the correlation coefficient between INP and chemical element mass concentration in batches.
# 批量绘制INP数浓度与化学元素质量浓度之间相关系数的热力图.
# Current version: v1.0

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import pearsonr

def configure_plot_style():
    """配置符合毕业论文要求的全局绘图样式"""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] 
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman'
    
def format_species_name(name):
    name = name.replace('PM2.5', 'PM$_{2.5}$')
    name = name.replace('NH4+', 'NH$_4^+$')
    name = name.replace('SO42-', 'SO$_4^{2-}$')
    name = name.replace('NO3-', 'NO$_3^-$')
    name = name.replace('Na+', 'Na$^+$')
    name = name.replace('Cl-', 'Cl$^-$')
    name = name.replace('Mg2+', 'Mg$^{2+}$')
    name = name.replace('N_INP(#/L)', 'N_INP')
    name = name.replace('Ca2+', 'Ca$^{2+}$')
    name = name.replace('K+', 'K$^{+}$')
    name = name.replace('NH3', 'NH$_3$')
    name = name.replace('HNO3', 'HNO$_3$')
    name = name.replace('SO2', 'SO$_2$')
    name = name.replace('HNO2', 'HNO$_2$')
    name = name.replace('n_s', 'n$_s$')
    name = name.replace('N_10-500nm', 'N$_{10-500nm}$')
    name = name.replace('N_10-1000nm', 'N$_{10-1000nm}$')
    name = name.replace('N_500-2500nm', 'N$_{500-2500nm}$')
    name = name.replace('N_1000-2500nm', 'N$_{1000-2500nm}$')
    return name

def calculate_corr_and_p(df, log_before_calculate=True):
    """
    计算两两物质的Pearson相关系数(R)和P值。
    无法计算的情况(如样本过少、方差为0)返回 NaN。
    """
    cols = [format_species_name(col) for col in df.columns]
    n = len(cols)
    
    # 预先创建用于存储 R 和 P 的全 NaN DataFrame
    corr_matrix = pd.DataFrame(np.nan, index=cols, columns=cols)
    p_matrix = pd.DataFrame(np.nan, index=cols, columns=cols)
    
    # 确保 df 中的所有数据都是数值类型，非数值的转换为 NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    # 将 -999(默认缺失值) 替换为 NaN
    df = df.replace(-999, np.nan)
    
    if log_before_calculate:
        df_clean = df.where(df > 0, np.nan)
        df = np.log(df_clean)
    
    for i in range(n):
        for j in range(i):  # 只需要下三角 (j < i)
            col1 = df.iloc[:, i]
            col2 = df.iloc[:, j]
            
            # 获取同时非空的有效数据对
            valid_mask = ~col1.isna() & ~col2.isna()
            v1 = col1[valid_mask]
            v2 = col2[valid_mask]
            
            # 条件判断：样本数太少(<3) 或 某列数据方差为0
            if len(v1) < 3 or np.var(v1) == 0 or np.var(v2) == 0:
                corr_matrix.iloc[i, j] = np.nan
                p_matrix.iloc[i, j] = np.nan
            else:
                r, p = pearsonr(v1, v2)
                corr_matrix.iloc[i, j] = r
                p_matrix.iloc[i, j] = p
                
    return corr_matrix, p_matrix

def plot_correlation_heatmap(corr_matrix, p_matrix, output_path='correlation_heatmap.png'):
    """
    根据相关系数矩阵和P值矩阵绘制下三角热力图。
    """
    # 生成掩膜(Mask)，隐藏上三角及对角线
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(20, 16), dpi=800)
    
    # 将背景设为纯白，用于体现无法计算的 NaN 空白格
    ax.set_facecolor('white')
    
    # 绘制热力图 (使用 coolwarm 表现蓝-白-红过渡)
    sns.heatmap(corr_matrix, 
                mask=mask, 
                cmap='coolwarm', 
                vmin=-1, vmax=1, 
                center=0, 
                square=True, 
                annot=True, 
                fmt=".2f", 
                annot_kws={"size": 7, "family": "Times New Roman"}, 
                cbar_kws={"shrink": 0.8}, 
                ax=ax, 
                linewidths=0.5,       # 格子之间的基础灰线宽度
                linecolor='gray')
    
    # 遍历下三角，如果 p < 0.05 则绘制加粗边框
    n_rows, n_cols = corr_matrix.shape
    for i in range(n_rows):
        for j in range(i): # 只遍历下三角
            p_val = p_matrix.iloc[i, j]
            if not np.isnan(p_val) and p_val < 0.05:
                # 绘制矩形：左下角坐标为(j, i)，宽1，高1
                rect = patches.Rectangle((j, i), 1, 1, 
                                         fill=False, 
                                         edgecolor='black', 
                                         linewidth=1.8,   # 边框加粗
                                         zorder=10)       # 保证框画在最顶层
                ax.add_patch(rect)
                
    # 调整X、Y轴标签字体和旋转角度
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', 
                       fontname='Times New Roman', size=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, 
                       fontname='Times New Roman', size=9)
    
    # 调整 Colorbar 字体
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    for t in cbar.ax.get_yticklabels():
        t.set_fontname('Times New Roman')
        
    # 设置标题
    plt.title(f"INP$_{{{temp}}}$({season}) vs. Elements Correlation Heatmap", 
              fontdict={'family': 'Times New Roman', 'size': 20, 'weight': 'bold'}, 
              pad=20)
    
    # 紧凑布局并高分辨率保存
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"热力图已保存至: {output_path}")

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 样式初始化
    configure_plot_style()
    
    # 2. 读取数据
    df_data = pd.read_csv(r"D:\Coding\Data\Lanzhou_chemical\Corr_heatmap.csv")
    
    # 3. 计算相关系数和P值矩阵
    TEMPERATURE_LIST = [-35, -30, -25, -20, -15]

    for temp in TEMPERATURE_LIST:
        temp_mask = df_data['T_a(degC)'] == temp
        for season in ['Spring', 'Summer', 'Autumn', 'Winter']:
            season_mask = df_data['Season'] == season

            df_data_masked = df_data[temp_mask & season_mask].copy()
            colnames_to_remove = ['T_a(degC)', 'datetime', 'Time', 'Season', 'Hg', 'Mo', 'Sc', 'Br', 'Te', 'Cs', 'Nb']
            df_data_masked.drop(columns=colnames_to_remove, inplace=True, errors='ignore')

            corr_df, p_df = calculate_corr_and_p(df_data_masked, log_before_calculate=True)
            # 4. 制图与输出
            plot_correlation_heatmap(corr_df, p_df, output_path=fr'D:\Coding\master0_2025\Lanzhou_aerosol\Correlation_Heatmap_{temp}degC_{season}.png')