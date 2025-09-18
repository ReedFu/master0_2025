### ICE-T 观测: INP浓度与温度关系图v2
### 读取多个 TXT 文件, 绘制散点图
### 不具有显著性的数据用x形状的散点表示
### 使用CVI得到的数据用蓝色表示, 未使用CVI的数据用红色表示
### 时间: 2025-08-28
### 作者：付弘宇

import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 创建图形
fig, ax = plt.subplots(dpi=500)
ax.set_yscale('log')
ax.set_xlabel('Temperature(℃)')
ax.set_ylabel('INP concentration(L^-3)')
ax.tick_params(which='both', top=True, bottom=True, left=True, right=True, direction='in')
ax.set_title('INP concentration vs Temperature')
#ax.set_xlim(-30,-12)
#ax.set_ylim(1,10**6)


data_dir = pathlib.Path(r"E:\data\ICE-T\INP")# 数据文件夹路径
csv_files = list(data_dir.glob('CSU*.txt'))# 获取所有 TXT 文件
columns = ['STRT_DATE/TIME','STOP_DATE/TIME','LAT','LON','ALT','CVI','CVIF',
'AER','AER_S','IN_RAW','IN_COR','IN_COR_S','IN_SIG','T','T_S','W_SAT','W_SAT_S','P','P_S']# 手动设置列名

for csv_file in csv_files:
    df = pd.read_csv(csv_file, names=columns, skiprows=23, sep='\s+')

    # 数据清洗
    df = df.replace(9999, np.nan)#将负值数据(文件中用9999表示)替换为NaN
    df = df.dropna(subset=['IN_COR', 'T'])

    # 数据分类
    cvi_df = df[df['CVI'] == 1]  # 使用CVI的数据
    noncvi_df = df[df['CVI'] == 0]  # 不使用CVI的数据

    # 进一步根据显著性分类
    sig_cvi_df = cvi_df[cvi_df['IN_SIG'] == 1]  # 使用CVI且具有显著性的数据
    nonsig_cvi_df = cvi_df[cvi_df['IN_SIG'] == 0]  # 使用CVI但不具有显著性的数据
    sig_noncvi_df = noncvi_df[noncvi_df['IN_SIG'] == 1]  # 不使用CVI但具有显著性的数据
    nonsig_noncvi_df = noncvi_df[noncvi_df['IN_SIG'] == 0]  # 不使用CVI且不具有显著性的数据

    # 绘制散点图
    ax.scatter(sig_cvi_df['T'], sig_cvi_df['IN_COR'], color='blue', s=30)
    ax.scatter(nonsig_cvi_df['T'], nonsig_cvi_df['IN_COR'], marker='x', color='blue', s=30)
    ax.scatter(sig_noncvi_df['T'], sig_noncvi_df['IN_COR'], color='red', s=30)
    ax.scatter(nonsig_noncvi_df['T'], nonsig_noncvi_df['IN_COR'], marker='x', color='red', s=30)

# 添加图例
ax.legend(['CVI&Sig', 'CVI&Non-Sig','Non-CVI&sig','Non-CVI&Non-Sig'], loc='best')

#plt.show()
plt.savefig("./master0_2025/ICE-T/INP_vs_Temperature_v2.png", bbox_inches='tight')