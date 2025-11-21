### INP浓度和降水量的时间序列 (2025.10)

## 导入相关库, 定义函数(用于数据处理)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

def extract_segment_last(s):
    """
    输入: 一个 Series 对象, 这个 Series 由'数据--NaN--数据'交替组成.

    返回: Series 对象, 包含每个测量段的最后一个值(认为数据趋于稳定后才有意义), 索引为这些值的原始索引.

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

    返回: Series 对象, 索引为两个输入 Series 索引的平均值.

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

    idx = np.round((bac.index + sam.index)/2).astype(int)
    net = sam.values - bac.values
    net[net < 0] = 0
    return pd.Series(net, index=idx) 

temperature = [-15, -20, -25, -30]
for i, temp in enumerate(temperature):
    ## 可视化数据的提取与处理
    ### 处理INP数据
    p = Path(r"D:\Coding\Data\Lanzhou_cfdc")
    all_dates = []
    all_INP = []
    count = 0
    ### 数据处理, 合并多天INP数据, 保存为列表格式
    for file in p.glob("data_202510*.csv"): #### 修改此处文件名可选择不同时间段的数据 ####
        data = pd.read_csv(file).dropna(subset=['Lamina Average T [C]'])
        #### 确定日期
        base_date = pd.to_datetime(file.stem.split('_')[1][0:8])
        dt = base_date + pd.to_timedelta(data['Time'], unit='s')
        
        bac_toggle = data['Background INP avg [#/L] Toggle']
        sam_toggle = data['Sample INP avg [#/L] Toggle']
        #### 一定温度下INP浓度提取, 修改此处温度值可选择不同温度下的INP浓度 ####
        N_inp_bac = data['Background INP avg [#/L]'][bac_toggle==True][data['Lamina T Set [C]']==temp].reindex(dt.index, fill_value=np.nan)
        N_inp_sam = data['Sample INP avg [#/L]'][sam_toggle==True][data['Lamina T Set [C]']==temp].reindex(dt.index, fill_value=np.nan)
        #### 计算净INP浓度
        N_inp_avg_net = compute_net(sam=N_inp_sam, bac=N_inp_bac)
        #### 保存为列表格式
        all_dates.append(dt[N_inp_avg_net.index])
        all_INP.append(N_inp_avg_net)
        #### 输出文件名以及该文件处理后的INP数据点数量
        #print(f"Processed file: {file.name}, INP_net count: {len(N_inp_avg_net)}")
        if len(N_inp_avg_net) != 0:
            count += 1
    print(f"There are {count} valid files with INP data.")

    ### 合并所有INP数据, 创建DataFrame(INP是其中一列)
    all_dates = pd.concat(all_dates, ignore_index=True)# 在这之后, 索引变得不重要了
    all_INP = pd.concat(all_INP, ignore_index=True)
    df_inp = pd.DataFrame({'Date': all_dates,
                    'N_inp_net': all_INP})
    ### 处理降水数据
    data_prp = pd.read_csv(r"D:\Coding\Data\Lanzhou_precipitation\rp5.ru\52983.01.02.2024.11.11.2025.1.0.0.en.utf8.00000000.csv"
        , skiprows=6
        , sep=";"
        , index_col=False
    )
    date = pd.to_datetime(data_prp['Local time in Lanzhou'], format='%d.%m.%Y %H:%M')
    #### 选择数据, 修改此处时间可选择**降水**时间序列的起止时间 ####
    data_prp_selected = data_prp[(date >= '2025-10-01') & (date < '2025-11-01')]
    ### 读取日期数据, 为绘图做准备
    x_inp = df_inp['Date']
    x_prp = pd.to_datetime(data_prp_selected['Local time in Lanzhou'], format='%d.%m.%Y %H:%M')
    ### 读取INP数据, 为绘图做准备
    y_inp = df_inp['N_inp_net']
    ### 读取降水数据, 为绘图做准备
    y_prp = data_prp_selected['RRR'].replace('No precipitation', 0).astype(float)

    ## 可视化绘图
    if i == 0:
        fig = plt.figure(dpi=1000, figsize=(16, 8))# 只创建一次画布
    ax1 = fig.add_subplot(2, 2, i+1)#子图位置

    ax1.bar(x_prp, y_prp, color='tab:blue', edgecolor='k', width=0.5)
    ax2 = ax1.twinx()
    ax2.scatter(x_inp, y_inp, color='tab:red', edgecolor='k')
    ### x轴的设置
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=3))# 每隔3天设置一个主刻度
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()# 自动旋转日期标记
    ax1.set_xlabel('Date in October 2025')###
    ### y轴的设置
    ax1.set_ylabel('Precipitation (mm/3h)', color='tab:blue')
    if (y_inp > 0).any():
        ax2.set_yscale('log')### 对数坐标
    ax2.set_ylabel(f'INP Concentration at {temp}℃(#/L)', color='tab:red')###
### 保存图片
plt.savefig(rf"D:\Coding\master0_2025\Lanzhou_cfdc\202510.png")###