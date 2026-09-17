# 将兰州空气质量监测站点的污染物数据(多个CSV文件)处理为单个CSV文件
# * 数据时间(Time Range of Data): 2024年1月1日-2025年12月31日(共两年数据)
# * 各数据变量的含义(Data Type): 请阅读文件夹中的README.md
# * 数据来源(Data Source): 全国城市空气质量实时发布平台(https://air.cnemc.cn:18007/)
# * 致谢(Acknowledgements): 感谢王晓磊提供数据爬取, 王晓磊的博客: https://quotsoft.net/air/
# * 目前版本: v1.0
# * 存在的问题: '1478A': 在 `2025-01-23 19:00:00` 之后数据缺失; '1479A': 在 `2025-10-01 21:00:00` 之后数据缺失. 认为是数据的问题, 而非代码的问题.

import pathlib
import pandas as pd
import numpy as np

# ======= 全局变量设置 =======

# 1. 选择处理的监测站点
STATION = '1479A' # 该监测站点位于兰州市铁路设计院, 距CFDC仪器约2.27公里
#STATION = '1478A' # 该监测站点位于兰州市生物制品所, 距CFDC仪器约3公里
# 2. 设置数据文件夹路径
DATA_PATH = pathlib.Path("D:/Coding/Data/Lanzhou_pollution/Xiaolei_Wang/all")
# 3. 设置保存路径(文件命名为 `{站点名}_data.csv` )
SAVE_PATH = pathlib.Path("D:/Coding/Data/Lanzhou_pollution/Xiaolei_Wang")

# ==========================

def main():
    all_time = []
    AQI = []
    PM25 = []
    PM25_24h = []
    PM10 = []
    PM10_24h = []
    SO2 = []
    SO2_24h = []
    NO2 = []
    NO2_24h = []
    O3 = []
    O3_24h = []
    O3_8h = []
    O3_8h_24h = []
    CO = []
    CO_24h = []
    for file in DATA_PATH.glob("*.csv"):
            df = pd.read_csv(file)
            date = pd.to_datetime(df['date'], format='%Y%m%d')
            hours = pd.to_timedelta(df['hour'], unit='h')
            time = date + hours
            df.index = time

            AQI.append(df[df['type']=='AQI'][STATION])
            PM25.append(df[df['type']=='PM2.5'][STATION])
            PM25_24h.append(df[df['type']=='PM2.5_24h'][STATION])
            PM10.append(df[df['type']=='PM10'][STATION])
            PM10_24h.append(df[df['type']=='PM10_24h'][STATION])
            SO2.append(df[df['type']=='SO2'][STATION])
            SO2_24h.append(df[df['type']=='SO2_24h'][STATION])
            NO2.append(df[df['type']=='NO2'][STATION])
            NO2_24h.append(df[df['type']=='NO2_24h'][STATION])
            O3.append(df[df['type']=='O3'][STATION])
            O3_24h.append(df[df['type']=='O3_24h'][STATION])
            O3_8h.append(df[df['type']=='O3_8h'][STATION])
            O3_8h_24h.append(df[df['type']=='O3_8h_24h'][STATION])
            CO.append(df[df['type']=='CO'][STATION])
            CO_24h.append(df[df['type']=='CO_24h'][STATION])
            all_time.append(pd.Series(list(set(time)))) # 避免重复时间
        
    all_time = pd.concat(all_time, ignore_index=True).sort_values().reset_index(drop=True)# 排序不能省略, 否则时间和其他数据对不上

    df_pollution = pd.DataFrame({'Date': all_time,
        'AQI': pd.concat(AQI, ignore_index=True),
        'PM2.5': pd.concat(PM25, ignore_index=True),
        'PM2.5_24h': pd.concat(PM25_24h, ignore_index=True),
        'PM10': pd.concat(PM10, ignore_index=True),
        'PM10_24h': pd.concat(PM10_24h, ignore_index=True),
        'SO2': pd.concat(SO2, ignore_index=True),
        'SO2_24h': pd.concat(SO2_24h, ignore_index=True),
        'NO2': pd.concat(NO2, ignore_index=True),
        'NO2_24h': pd.concat(NO2_24h, ignore_index=True),
        'O3': pd.concat(O3, ignore_index=True),
        'O3_24h': pd.concat(O3_24h, ignore_index=True),
        'O3_8h': pd.concat(O3_8h, ignore_index=True),
        'O3_8h_24h': pd.concat(O3_8h_24h, ignore_index=True),
        'CO': pd.concat(CO, ignore_index=True),
        'CO_24h': pd.concat(CO_24h, ignore_index=True),
        })

    df_pollution.to_csv(SAVE_PATH / f"{STATION}_data.csv", index=False)

if __name__ == "__main__":
    main()