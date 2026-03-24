# -*- coding: utf-8 -*-
# 将全年CFDC数据输出为单个CSV文件
# 目前版本: v2.4
# 版本记录:
# v2.0: 在v1.0的基础上, 修改了数据处理的思路, 实现: 采样段前后的背景段取平均, 作为这部分采样段的背景值. (如果仅有前背景段, 则取前背景段作为采样段的背景值.)
# v2.1: 修改了温度和过饱和度的计算方式, 采用平均值而非最后一个值.
# v2.2, 增加数据质量控制步骤, 去除异常浓度值(包括0值和负值, SS_w<=4或SS_w>=6的值, 2024年9月17日之前未经验证的值)等; 增加"活化温度"列.
# v2.3, 增加"总采样体积"列, 对INP浓度进行初步的显著性检验(剔除INP浓度小于0.1 #/L的值).
# v2.4, 增加对 INP 浓度的显著性检验(Schill et al., 2016; DeMott et al., 2017): 1. 根据泊松分布计算采样段和背景段的 INP 浓度的标准差; 2. 两标准差的平方和作为 INP 净浓度的误差; 3. 大于 INP 净浓度误差的 1.64 倍才认为是显著的数据点(Z statistic at 95% confidence).

import pandas as pd
import numpy as np
from pathlib import Path

# ======= 全局变量设置 =======

TEMPERATURE = [-15, -20, -25, -30, -35] # 需要处理的温度列表, 单位: °C
START_TIME = pd.to_datetime("2024-09-17 00:00:00") # 数据处理的起始时间, 包含在内
END_TIME = pd.to_datetime("2025-10-01 00:00:00") # 数据处理的结束时间, 包含在内
CFDC_PATH = Path(r"D:\Coding\Data\Lanzhou_cfdc") # CFDC数据文件(csv格式)所在路径, 不包含子文件夹

# ==========================

def sum_by_consecutive_index(s: pd.Series) -> pd.Series:
    '''
    Docstring for sum_by_consecutive_index
    
    :param s: Description: 输入的 Series 对象, 索引为整数, 有几小段索引连续的值, 但这几段之间不连续. 例如, 索引为 [1,2,3,4,5,11,12,13,14,15] 的 Series.
    :type s: pd.Series
    :return: Description: 输出为 Series 对象, 将每段连续索引的值分组求和, 每组的索引为每段的终点索引.
    :rtype: Series[Any]
    '''
    #if s.empty:
        #return s.copy()

    # 确保索引已排序
    # s = s.sort_index()

    tag = s.index.to_series().diff().ne(1).cumsum()

    # 1) 每段求和
    out = s.groupby(tag, sort=False).sum()

    # 2) 每段终点索引
    end_idx = s.index.to_series().groupby(tag, sort=False).max()

    out.index = end_idx.to_numpy()
    #out.index.name = s.index.name  # 可选：保留索引名
    return out

def sum_with_next(s: pd.Series) -> pd.Series:
    '''
    Docstring for sum_with_next
    
    :param s: Description: 输入的 Series 对象, 索引为整数(但无需从0或1开始).
    :type s: pd.Series
    :return: Description: 输出为 Series 对象, 每个索引的值等于原始值加上下一个索引的值. 最后一个索引的值保持不变.
    :rtype: Series[Any]
    '''
    for i in range(len(s.index)):
        if i+1 == len(s.index): # 如果i是最后一个索引
            s.iloc[i] = s.iloc[i]
            break
        s.iloc[i] = s.iloc[i] + s.iloc[i+1]
        
    return s 

def mean_by_consecutive_index(s: pd.Series) -> pd.Series:
    '''
    Docstring for mean_by_consecutive_index
    
    :param s: Description: 输入的 Series 对象, 索引为整数, 有几小段索引连续的值, 但这几段之间不连续. 例如, 索引为 [1,2,3,4,5,11,12,13,14,15] 的 Series.
    :type s: pd.Series
    :return: Description: 输出为 Series 对象, 将每段连续索引的值分组求平均, 每组的索引为每段的终点索引.
    :rtype: Series[Any]
    '''
    #if s.empty:
        #return s.copy()

    # 确保索引已排序
    # s = s.sort_index()

    tag = s.index.to_series().diff().ne(1).cumsum()

    # 1) 每段求平均
    out = s.groupby(tag, sort=False).mean()

    # 2) 每段终点索引
    end_idx = s.index.to_series().groupby(tag, sort=False).max()

    out.index = end_idx.to_numpy()
    #out.index.name = s.index.name  # 可选：保留索引名
    return out

def process_different_length(sam: pd.Series, bac: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    输入: 两个长度不同的 Series 对象(索引为整数), 每个 Series 的值都属于同一类别(采样段或背景段).

    返回: 两个长度相同的 Series 对象(索引为整数), 处理方式: 删除连续两段(同属采样段或背景段)的前一段.

    调用此函数前, 需要:
    - import pandas as pd
    """
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
    #### 验证确已删除
    #print("After processing:")
    #print("Background: \n", bac)
    #print("Sample: \n", sam)

    return sam, bac

def compute_net(sam, bac):
    """
    输入: 两个 Series 对象(索引为整数), 计算净值 (Sample - Background), 如果净值小于 0, 则取 0.

    返回: Series 对象, 索引为采样(sample) Series 的索引.

    调用此函数前, 需要:
    - import pandas as pd
    """
    
    ### 保证两个 Series 长度相同
    if len(sam) != len(bac):
        #### 调试输出
        #print("Sample and background have different number of measurement segments.")

        sam, bac = process_different_length(sam, bac)

        if len(sam) != len(bac):
            raise ValueError("After processing, Sample and Background still have different number of measurement segments.")
    
    net = sam.values - bac.values
    #net[net < 0] = 0
    return pd.Series(net, index=sam.index)

def compute_sigma_net(sigma_sam, sigma_bac):
    """
    输入: 两个 Series 对象(索引为整数), 分别为采样段和背景段的标准差, 计算净浓度的标准差 (sqrt(sigma_sam^2 + sigma_bac^2)).

    返回: Series 对象, 索引为采样(sample) Series 的索引.

    调用此函数前, 需要:
    - import pandas as pd
    """
    
    ### 保证两个 Series 长度相同
    if len(sigma_sam) != len(sigma_bac):
        #### 调试输出
        #print("Sample and background have different number of measurement segments.")

        sigma_sam, sigma_bac = process_different_length(sigma_sam, sigma_bac)

        if len(sigma_sam) != len(sigma_bac):
            raise ValueError("After processing, Sample and Background still have different number of measurement segments.")
        
    sigma_net = np.sqrt(sigma_sam.values**2 + sigma_bac.values**2)

    return pd.Series(sigma_net, index=sigma_sam.index)

def main():
    all_dates = []
    all_INP = []
    all_T = []
    all_SS_w = []
    all_SS_i = []
    all_vol = []
    all_sig_level = []
    all_significant = []
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
            if data.empty:
                continue
            #### 确定日期
            seconds = pd.to_numeric(data['Time'], errors='coerce')# 把非数字变成 NaN (这行用于处理非utf-8编码时出现的异常)
            time = base_date + pd.to_timedelta(seconds, unit='s')
            
            bac_toggle = data['Background INP avg [#/L] Toggle']
            sam_toggle = data['Sample INP avg [#/L] Toggle']
            #### 自行计算INP平均浓度(而不是直接使用仪器输出的平均浓度)
            #### 先算背景段平均浓度: 计算这一天多个背景段的浓度, 每个背景段得到一个INP平均浓度, 保存于 Series 中
            num_inp_bac = data['INP Counts'][bac_toggle==True][data['Lamina T Set [C]']==temp]
            total_flow_bac = data['Total Mass Flow [SLPM]'][bac_toggle==True][data['Lamina T Set [C]']==temp]
            sheath_flow_bac = data['Sheath Mass Flow [SLPM]'][bac_toggle==True][data['Lamina T Set [C]']==temp]
            aerosol_flow_bac = total_flow_bac - sheath_flow_bac # unit: standard L per minute

            count_bac = sum_by_consecutive_index(num_inp_bac) # unit of `num_inp_bac`: # (per second); unit of `count_bac`: #
            vol_bac = sum_by_consecutive_index(aerosol_flow_bac) / 60  # unit of `aerosol_flow_bac`:standard L (per minute); unit of `vol_bac`: standard L
            conc_inp_bac = sum_with_next(count_bac) / sum_with_next(vol_bac) # unit: #/standard L

            #### 再算采样段平均浓度: 计算这一天多个采样段的浓度, 每个采样段得到一个INP平均浓度, 保存于 Series 中
            num_inp_sam = data['INP Counts'][sam_toggle==True][data['Lamina T Set [C]']==temp]
            total_flow_sam = data['Total Mass Flow [SLPM]'][sam_toggle==True][data['Lamina T Set [C]']==temp]
            sheath_flow_sam = data['Sheath Mass Flow [SLPM]'][sam_toggle==True][data['Lamina T Set [C]']==temp]
            aerosol_flow_sam = total_flow_sam - sheath_flow_sam # unit: standard L per minute

            count_sam = sum_by_consecutive_index(num_inp_sam) # unit of `num_inp_sam`: # (per second); unit of `count_sam`: #
            vol_sam = sum_by_consecutive_index(aerosol_flow_sam) / 60  # unit of `aerosol_flow_sam`:standard L (per minute); unit of `vol_sam`: standard L
            conc_inp_sam = count_sam / vol_sam # unit: #/standard L

            T_inp = data['Lamina Average T [C]'][sam_toggle==True][data['Lamina T Set [C]']==temp]
            T_inp = mean_by_consecutive_index(T_inp)
            SS_w = data['Lamina SS_w'][sam_toggle==True][data['Lamina T Set [C]']==temp]
            SS_w = mean_by_consecutive_index(SS_w)
            SS_i = data['Lamina SS_i'][sam_toggle==True][data['Lamina T Set [C]']==temp]
            SS_i = mean_by_consecutive_index(SS_i)
            #### 计算净INP浓度
            N_inp_avg_net = compute_net(sam=conc_inp_sam, bac=conc_inp_bac)
            #### 计算显著性水平
            sigma_inp_bac = np.sqrt(conc_inp_bac / vol_bac) # 根据泊松分布, INP 浓度的标准差为 sqrt(计数) / 体积 = sqrt(浓度 * 体积) / 体积 = sqrt(浓度 / 体积)
            sigma_inp_sam = np.sqrt(conc_inp_sam / vol_sam) 
            sigma_inp_net = compute_sigma_net(sigma_sam=sigma_inp_sam, sigma_bac=sigma_inp_bac)
            sig_level = 1.64 * sigma_inp_net
            is_significant = N_inp_avg_net > sig_level
            #### 保存为列表格式
            all_dates.append(time[N_inp_avg_net.index])
            all_INP.append(N_inp_avg_net)
            all_T.append(T_inp[N_inp_avg_net.index])
            all_SS_w.append(SS_w[N_inp_avg_net.index])
            all_SS_i.append(SS_i[N_inp_avg_net.index])
            all_vol.append(vol_sam[N_inp_avg_net.index])
            all_sig_level.append(sig_level[N_inp_avg_net.index])
            all_significant.append(is_significant[N_inp_avg_net.index])
            #### 输出文件名以及该文件处理后的INP数据点数量
            #print(f"Processed file: {file.name}, INP_net count: {len(N_inp_avg_net)}")
            if len(N_inp_avg_net) != 0:
                count += 1
        print(f"There are {count} valid files with INP{temp} data.")

    all_dates = pd.concat(all_dates, ignore_index=True)# 在这之后, 索引变得不重要了
    all_INP = pd.concat(all_INP, ignore_index=True)
    all_T = pd.concat(all_T, ignore_index=True)
    all_SS_w = pd.concat(all_SS_w, ignore_index=True)
    all_SS_i = pd.concat(all_SS_i, ignore_index=True)
    all_vol = pd.concat(all_vol, ignore_index=True)
    all_sig_level = pd.concat(all_sig_level, ignore_index=True)
    all_significant = pd.concat(all_significant, ignore_index=True)
    df_inp = pd.DataFrame({'Date': all_dates,
        'N_inp_net(#/L)': all_INP,
        'T_inp(°C)': all_T
        ,'SS_w': all_SS_w
        ,'SS_i': all_SS_i
        ,'Total_Sampling_Volume(L)': all_vol
        ,'Significance_Level(#/L)': all_sig_level
        ,'Is_Significant': all_significant})
    df_inp = df_inp.sort_values('Date').reset_index(drop=True)

    # 质量控制
    print(f"质量控制前, 点的数量: {df_inp.shape[0]}")

    # 1. 去除正的温度值
    df_inp = df_inp.where(df_inp['T_inp(°C)'] < 0).dropna()
    print(f"去除正温度值后, 点的数量: {df_inp.shape[0]}")

    # 2. 去除非正的INP浓度值
    df_inp = df_inp[df_inp['N_inp_net(#/L)'] > 0]
    print(f"去除非正INP浓度值后, 点的数量: {df_inp.shape[0]}")
    
    # 3. 过饱和度控制在合理范围内 (4 < SS_w < 6)
    df_inp = df_inp[(df_inp['SS_w'] > 4) & (df_inp['SS_w'] < 6)]
    print(f"过饱和度控制后, 点的数量: {df_inp.shape[0]}")

    # 4. 显著性检验:
    print(f"显著性检验后, 点的数量: {df_inp[df_inp['Is_Significant']].shape[0]} (占比: {df_inp[df_inp['Is_Significant']].shape[0]/df_inp.shape[0]:.2%})")

    # 5. 根据日期归属, 增加“季节”列
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

    # 6. 增加"活化温度"列
    def activation_temperature(temp: pd.Series):
        if abs(temp - (-15)) < 2.5:
            return -15
        elif abs(temp - (-20)) < 2.5:
            return -20
        elif abs(temp - (-25)) < 2.5:
            return -25
        elif abs(temp - (-30)) < 2.5:
            return -30
        elif abs(temp - (-35)) < 2.5:
            return -35
        else:
            return np.nan
    df_inp['T_a(°C)'] = df_inp['T_inp(°C)'].apply(activation_temperature)

    # 输出为CSV文件
    df_inp.to_csv(r"D:\Coding\Data\Lanzhou_cfdc\processed\N_INP(202409-202509)v2.4.csv", index=False)


if __name__ == "__main__":
    main()