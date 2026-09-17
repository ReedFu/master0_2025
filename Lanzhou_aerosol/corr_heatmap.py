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
    name = name.replace('OC(optical)', 'OC')
    name = name.replace('EC(optical)', 'EC')
    #name = name.replace('Mineral dust', 'MD')
    #name = name.replace('Vehicle emissions', 'VE')
    #name = name.replace('Secondary formation', 'SF')
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
                annot_kws={"size": 20, "family": "Times New Roman"}, 
                cbar_kws={"shrink": 0.8}, 
                ax=ax, 
                linewidths=0.5,
                linecolor='gray')
    
    # 遍历下三角，如果 p < 0.05 则在格子右上角添加显著性标记
    n_rows, n_cols = corr_matrix.shape
    for i in range(n_rows):
        for j in range(i): # 只遍历下三角
            p_val = p_matrix.iloc[i, j]
            if not np.isnan(p_val) and p_val < 0.05:
                star = '*'
                ax.text(j + 0.9, i + 0.25, star,
                        ha='center', va='center', fontsize=20, fontfamily='Times New Roman')
    
    # 调整X、Y轴标签字体和旋转角度
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', 
                       fontname='Times New Roman', size=20)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, 
                       fontname='Times New Roman', size=20)
    
    # 调整 Colorbar 字体
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    for t in cbar.ax.get_yticklabels():
        t.set_fontname('Times New Roman')
        
    # 设置标题
    plt.title(f"INP$_{{{target_temp}}}$({season}) vs. Elements Correlation Heatmap", 
              fontdict={'family': 'Times New Roman', 'size': 20, 'weight': 'bold'}, 
              pad=20)
    
    # 紧凑布局并高分辨率保存
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"热力图已保存至: {output_path}")

def plot_scatter(df, x_col, y_col, season, output_path='scatter_plot.png'):
    """
    绘制两个物质的散点图，并在图中显示 Pearson 相关系数和 P 值。
    """
    # 过滤有效数据
    df_clean = df[[x_col, y_col]].dropna()
    
    # 计算相关系数和P值
    if len(df_clean) < 3 or np.var(df_clean[x_col]) == 0 or np.var(df_clean[y_col]) == 0:
        r, p = np.nan, np.nan
    else:
        r, p = pearsonr(df_clean[x_col], df_clean[y_col])
    
    # 创建散点图
    fig , ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.scatter(df_clean[x_col], df_clean[y_col], alpha=0.7)
    
    # 添加标题和标签
    ax.set_title(f"{season}", 
                 fontdict={'family': 'Times New Roman', 'size': 16})
    ax.set_xlabel(format_species_name(x_col) + ' ($\mu g/m^3$)', fontname='Times New Roman', fontsize=14)
    ax.set_ylabel(format_species_name(y_col) + ' ($\mu g/m^3$)', fontname='Times New Roman', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # 显示相关系数和P值
    if np.isnan(r) or np.isnan(p):
        ax.text(0.05, 0.95, "Pearson r = NaN\np = NaN", 
                transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', fontname='Times New Roman')
    elif p < 0.05:
        ax.text(0.05, 0.95, f"Pearson r = {r:.2f}\np < 0.05", 
                transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', fontname='Times New Roman')
    else:
        ax.text(0.05, 0.95, f"Pearson r = {r:.2f}\np > 0.05", 
                transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', fontname='Times New Roman')
    
    # 紧凑布局并保存
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"散点图已保存至: {output_path}")
# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 样式初始化
    configure_plot_style()
    
    # 2. 读取数据
    df_data = pd.read_csv(r"D:\Coding\Data\Lanzhou_chemical\Corr_heatmap_merged.csv")
    
    # 3. 计算相关系数和P值矩阵
    # TEMPERATURE_LIST = [-35, -30, -25, -20, -15]
    target_temp = -30
    
    temp_mask = df_data['T_a(degC)'] == target_temp
    for season in ['Spring', 'Summer', 'Autumn', 'Winter']:
        season_mask = df_data['Season'] == season

        df_data_masked = df_data[temp_mask & season_mask].copy()
        colnames_to_keep = [
            "N_INP(#/L)",
            "n_s",
            "Mineral dust",
            "Vehicle emissions",
            "Secondary formation",
            "OC(optical)",
            "EC(optical)",
            "Ba",
            "Cu",
            "Ca",
            "K",
            "Si",
            "Mn",
            "Fe",
            "NH4+",
            "SO42-",
            "NO3-",
            "N_10-500nm",
            "N_10-1000nm",
            "N_500-2500nm",
            "N_1000-2500nm",
        ]
        df_data_masked = df_data_masked[colnames_to_keep]

        corr_df, p_df = calculate_corr_and_p(df_data_masked, log_before_calculate=False)
        # 4. 制图与输出
        plot_correlation_heatmap(corr_df, p_df, output_path=fr'D:\Coding\master0_2025\Thesis\Correlation_Heatmap_{target_temp}degC_{season}.png')

        # 选取两个感兴趣的物质进行散点图绘制
        # 例如: Mineral dust vs. Secondary formation
        plot_scatter(df_data_masked, 'Mineral dust', 'Secondary formation', season, output_path=fr'D:\Coding\master0_2025\Thesis\Scatter_MineralDust_SecondaryFormation_{target_temp}degC_{season}.png')