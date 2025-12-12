# -*- coding: utf-8 -*-
# 将全年数据输出为单个CSV文件
# 目前版本: v1.0
# 局限性: 背景段的INP平均浓度仅考虑前一背景段, 无法考虑采样段后一背景段的影响(无法计算前后两背景段INP平均浓度)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======= 全局变量设置 =======

TEMPERATURE = [-15, -20, -25, -30, -35]
START_TIME = pd.to_datetime("2024-09-01 00:00:00")
END_TIME = pd.to_datetime("2025-10-01 00:00:00")
CFDC_PATH = Path(r"D:\Coding\Data\Lanzhou_cfdc")

# ==========================

def extract_segment_last(s):
    """
    目的: 提取每个测量段(背景段/采样段)的最后一个平均值.

    输入: 一个 Series 对象, 这个 Series 由'数据--NaN--数据'交替组成.(这里的所谓'数据', 指的是某个测量段的累计平均值)

    返回: Series 对象, 包含每个测量段的最后一个值(认为累计平均值趋于稳定后才有意义), 索引为这些值的原始索引.

    调用此函数前, 需要:
    - import numpy as np
    - import pandas as pd
    """
    ### 保证输入是 Series
    if not isinstance(s, pd.Series):
        raise TypeError("Input must be a pandas Series.")
    
    ### 1. 提取每段非 NaN 区间的尾值
    mask = s.notna() & s.shift(-1).isna()
    s_last = s[mask]

    ### 2. 如果没有尾值，返回空 Series
    if len(s_last) == 0:
        return pd.Series(dtype=float)
    
    return s_last


def compute_net(sam, bac):
    """
    输入: 两个 Series 对象(索引为整数), 计算净值 (Sample - Background), 如果净值小于 0, 则取 0.

    返回: Series 对象, 索引为采样(sample) Series 的索引.

    调用此函数前, 需要:
    - import numpy as np
    - import pandas as pd
    - 定义 extract_segment_last 函数, 用于获取每个测量段的最后一个值
    """

    sam = extract_segment_last(sam)
    bac = extract_segment_last(bac)

    ### 保证两个 Series 长度相同
    if len(sam) != len(bac):
        #### 调试输出
        #print("Sample and background have different number of measurement segments.")
        
        ### 如果长度不一致, 尝试进行处理: 
        df1 = pd.DataFrame({'value': sam, 'tag': 'Sample'})
        df2 = pd.DataFrame({'value': bac, 'tag': 'Background'})
        df = pd.concat([df1, df2]).sort_index()
        #### 调试输出, 用于验证删除是否正确
        #print("Before processing:")
        #print(df)
        #### 删除连续两段数据的前一段 (即, 若连续两段数据属于同一tag, 保留后一段)
        df = df.drop(df[df['tag'] == df['tag'].shift(-1)].index)
        sam = df[df['tag'] == 'Sample']['value']
        bac = df[df['tag'] == 'Background']['value']

        ### 对于两种特殊情况的处理: 背景在最后; 样品在最前
        if df['tag'].iloc[-1]=='Background':
            bac = bac[:-1]
        if df['tag'].iloc[0]=='Sample':
            sam = sam[1:]
        ### 验证确已删除
        #print("After processing:")
        #print("Background: \n", bac)
        #print("Sample: \n", sam)

        if len(sam) != len(bac):
            raise ValueError("After processing, Sample and Background still have different number of measurement segments.")
    
    net = sam.values - bac.values
    net[net < 0] = 0
    return pd.Series(net, index=sam.index) 

def main():
    all_dates = []
    all_INP = []
    all_T = []
    all_SS_w = []
    all_SS_i = []
    for temp in TEMPERATURE:
        ## 可视化数据的提取与处理
        ### 处理INP数据
        
        count = 0
        ### 数据处理, 合并多天INP数据, 保存为列表格式
        for file in CFDC_PATH.glob("*.csv"): 
            base_date = pd.to_datetime(file.stem.split('_')[1][0:8])
            if not (START_TIME <= base_date <= END_TIME):
                continue
            data = pd.read_csv(file, encoding='utf-8', encoding_errors='ignore').dropna(subset=['Lamina Average T [C]'])
            #### 确定日期
            seconds = pd.to_numeric(data['Time'], errors='coerce')# 把非数字变成 NaN (这行用于处理非utf-8编码时出现的异常)
            time = base_date + pd.to_timedelta(seconds, unit='s')
            
            bac_toggle = data['Background INP avg [#/L] Toggle']
            sam_toggle = data['Sample INP avg [#/L] Toggle']
            #### 一定温度下INP浓度提取, 修改此处温度值可选择不同温度下的INP浓度 ####
            N_inp_bac = data['Background INP avg [#/L]'][bac_toggle==True][data['Lamina T Set [C]']==temp].reindex(time.index, fill_value=np.nan)
            N_inp_sam = data['Sample INP avg [#/L]'][sam_toggle==True][data['Lamina T Set [C]']==temp].reindex(time.index, fill_value=np.nan)
            T_inp = data['Lamina Average T [C]'][sam_toggle==True][data['Lamina T Set [C]']==temp].reindex(time.index, fill_value=np.nan)
            SS_w = data['Lamina SS_w'][sam_toggle==True][data['Lamina T Set [C]']==temp].reindex(time.index, fill_value=np.nan)
            SS_i = data['Lamina SS_i'][sam_toggle==True][data['Lamina T Set [C]']==temp].reindex(time.index, fill_value=np.nan)
            #### 计算净INP浓度
            N_inp_avg_net = compute_net(sam=N_inp_sam, bac=N_inp_bac)
            #### 保存为列表格式
            all_dates.append(time[N_inp_avg_net.index])
            all_INP.append(N_inp_avg_net)
            all_T.append(T_inp[N_inp_avg_net.index])
            all_SS_w.append(SS_w[N_inp_avg_net.index])
            all_SS_i.append(SS_i[N_inp_avg_net.index])
            #### 输出文件名以及该文件处理后的INP数据点数量
            #print(f"Processed file: {file.name}, INP_net count: {len(N_inp_avg_net)}")
            if len(N_inp_avg_net) != 0:
                count += 1
        print(f"There are {count} valid files with INP data.")

    all_dates = pd.concat(all_dates, ignore_index=True)# 在这之后, 索引变得不重要了
    all_INP = pd.concat(all_INP, ignore_index=True)
    all_T = pd.concat(all_T, ignore_index=True)
    all_SS_w = pd.concat(all_SS_w, ignore_index=True)
    all_SS_i = pd.concat(all_SS_i, ignore_index=True)
    df_inp = pd.DataFrame({'Date': all_dates,
        'N_inp_net(#/L)': all_INP,
        'T_inp(°C)': all_T
        ,'SS_w': all_SS_w
        ,'SS_i': all_SS_i})
    df_inp = df_inp.sort_values('Date').reset_index(drop=True)

    # 质量控制
    # 1. 去除正的温度值
    df_inp = df_inp.where(df_inp['T_inp(°C)'] < 0).dropna()

    # 2. 根据日期归属生成“季节”列
    def season_of_month(m):
        # 气象学季节：春3–5，夏6–8，秋9–11，冬12,1,2
        if m in [3,4,5]:
            return 'Spring'
        elif m in [6,7,8]:
            return 'Summer'
        elif m in [9,10,11]:
            return 'Autumn'
        elif m in [12,1,2]:
            return 'Winter'
        else:
            return 'Unknown'
    df_inp['season'] = df_inp['Date'].dt.month.map(season_of_month)

    # 输出为CSV文件
    df_inp.to_csv(r"D:\Coding\master0_2025\Lanzhou_cfdc\N_INP(202409-202509)(2).csv", index=False)


if __name__ == "__main__":
    main()