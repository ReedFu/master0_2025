# -*- coding: utf-8 -*-
# 主要功能: 读取并合并 SMPS 和 APS 的粒径分布数据, 计算总表面积浓度, 并与大气冰核 (INP) 数据进行时间对齐和表面活性密度 (ns) 计算.
# 目前版本: v1.0.2
# 版本记录:
# v1.0.1: 更新 INP 数据集为 v2.4.1 版本, 修正了 INP 数据中的异常值, 提高了数据质量. 
# v1.0.1(special version, SP): `MAX_APS_DP` 参数调整为 20000 nm, 以包含更大粒径范围的 APS 数据, 便于绘制气溶胶粒子谱分布.
# v1.0.2: 更新 INP 数据集为 v2.4.1 版本(修改部分列名和单位).

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import islice

# ==========================================
# 1. 数据加载与清理模块
# ==========================================

def read_instrument_export(path: Path, instr_type: str, encoding="cp1252", scan_lines=100) -> pd.DataFrame:
    """读取并解析单个仪器的 TXT 数据文件"""
    # 扫描前 scan_lines 行以寻找表头和分隔符
    with open(path, "r", encoding=encoding, errors="replace") as f:
        lines = list(islice(f, scan_lines))
        
    header_idx = None
    sep = "\t"
    for i, line in enumerate(lines):
        if "Sample #" in line and "Date" in line and "Start Time" in line:
            delims = {"\t": line.count("\t"), ",": line.count(","), ";": line.count(";")}
            sep = max(delims, key=delims.get)
            header_idx = i
            break
            
    if header_idx is None:
        raise ValueError(f"{path.name}: 前 {scan_lines} 行未找到表头")
        
    df = pd.read_csv(path, sep=sep, encoding=encoding, skiprows=header_idx, engine="python")
    
    # 根据仪器设定参数
    if instr_type == "SMPS":
        time_format = '%Y/%m/%d %H:%M:%S'
        bin_start_idx, bin_end_idx = 9, 112
    else: # APS
        time_format = "%m/%d/%y %H:%M:%S"
        bin_start_idx, bin_end_idx = 5, 56
        
    # 解析时间
    datetime_str = df["Date"].astype(str).str.strip() + " " + df["Start Time"].astype(str).str.strip()
    t = pd.to_datetime(datetime_str, errors="coerce", format=time_format)
    
    df = df.loc[t.notna()].copy()
    df["__dt__"] = t.loc[t.notna()].values
    
    # 提取粒径数据列并校验
    bin_cols = list(df.columns[bin_start_idx:bin_end_idx])
    try:
        _ = np.array(bin_cols).astype(float)
    except ValueError:
        raise ValueError(f"{path.name}: 粒径列选取错误, 无法转为浮点数 -> {bin_cols[:5]}")

    return df[["__dt__"] + bin_cols].copy()

def load_all_files(directory: str, instr_type: str) -> pd.DataFrame:
    """批量加载指定目录下的所有TXT文件并合并去重"""
    files = list(Path(directory).glob('*.TXT'))
    print(f"[{instr_type}] 正在加载 {len(files)} 个文件...")
    
    all_dfs = []
    for f in files:
        try:
            all_dfs.append(read_instrument_export(f, instr_type))
        except Exception as e:
            print(f"❌ 读取 {f.name} 失败: {e}")
            
    df = pd.concat(all_dfs, ignore_index=True)
    # 按时间排序并去重
    df = df.sort_values("__dt__").drop_duplicates(subset="__dt__", keep="last").reset_index(drop=True)
    return df

def remove_bad_periods(df: pd.DataFrame, bad_periods: list, time_col="__dt__") -> pd.DataFrame:
    """剔除指定异常的连续时间段"""
    if not bad_periods: return df
    mask = pd.Series(False, index=df.index)
    for start_str, end_str in bad_periods:
        start, end = pd.to_datetime(start_str), pd.to_datetime(end_str)
        mask |= (df[time_col] >= start) & (df[time_col] <= end)
    return df.loc[~mask].reset_index(drop=True)

# ==========================================
# 2. 预处理与合并模块
# ==========================================

def clean_and_sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将 DataFrame 列名转换为浮点数并升序排列"""
    new_cols = []
    for col in df.columns:
        try:
            new_cols.append(float(col))
        except ValueError:
            new_cols.append(col)
    df.columns = new_cols
    num_cols = [c for c in df.columns if isinstance(c, (int, float))]
    return df[num_cols].reindex(sorted(num_cols), axis=1)

def convert_aps_to_dm(aps_df: pd.DataFrame, rho_eff=1.5, rho_0=1.0, x=1.0) -> pd.DataFrame:
    """
    将 APS 空气动力学直径(Da) 转换为 电迁移率直径(Dm), 并修正单位为 nm
    
    rho_eff: 假设的有效密度 (g/cm3)

    rho_0: 标准密度 (1.0 g/cm3)

    x: 形状修正因子 (shape factor = 1.0, 表示球形粒子)
    """
    # 如果检测到第一个数值小于 20, 视作单位是微米(μm), 则转换为纳米(nm)
    if float(aps_df.columns[0]) < 20:
        aps_df.columns = [float(c) * 1000 for c in aps_df.columns]
    
    # 物理等式转换 Da -> Dm
    da = np.array(aps_df.columns)
    dm = da * np.sqrt(rho_0 * x / rho_eff)
    aps_df.columns = dm
    
    # 清洗排序
    aps_df = clean_and_sort_columns(aps_df)
    return aps_df

def merge_smps_aps(smps_df: pd.DataFrame, aps_df: pd.DataFrame, max_aps_dp=2500) -> pd.DataFrame:
    """拼接 SMPS 和 APS, 处理重叠区域以 SMPS 为准"""
    smps_max_dp = smps_df.columns.max()
    # 过滤重叠处 APS 数据，并切除过大的尾部部分
    aps_filtered = aps_df.loc[:, (aps_df.columns > smps_max_dp) & (aps_df.columns <= max_aps_dp)]
    
    # 索引均为时间戳，外连接合并
    combined = pd.concat([smps_df, aps_filtered], axis=1, join='outer')
    return clean_and_sort_columns(combined)

# ==========================================
# 3. 特征计算与外部分析集成模块
# ==========================================

def calculate_surface_area(df: pd.DataFrame) -> pd.Series:
    """计算气溶胶总表面积浓度 (μm^2/cm^3)"""
    dp_nm = np.array(df.columns.astype(float))
    dp_um = dp_nm / 1000.0
    
    # 计算 dlogDp (大多数仪器粒径对数等距)
    log_dp = np.log10(dp_nm)
    delta_log_dp = np.zeros_like(log_dp)
    delta_log_dp[:-1] = np.diff(log_dp)
    delta_log_dp[-1] = delta_log_dp[-2]
    
    # 表面积贡献 dS = pi * Dp^2 * (dN/dlogDp) * delta_log_dp
    multiplier = np.pi * (dp_um**2) * delta_log_dp
    surface_area_conc = df.multiply(multiplier, axis=1).sum(axis=1, min_count=1)
    
    return pd.Series(surface_area_conc, index=df.index, name='Total_Surface_Area_um2_cm3')

def get_instrument_status(combined_df: pd.DataFrame, missing_threshold=0.9) -> pd.DataFrame:
    """评估各设备通道有效数据的比例，返回仪器运行状态布尔表"""
    cols_smps = [col for col in combined_df.columns if float(col) <= 533.0]
    cols_aps  = [col for col in combined_df.columns if float(col) >= 549.0]
    
    status_smps = (combined_df[cols_smps].isna().sum(axis=1) / len(cols_smps)) <= missing_threshold
    status_aps  = (combined_df[cols_aps].isna().sum(axis=1) / len(cols_aps)) <= missing_threshold
    
    return pd.DataFrame({'Date': combined_df.index, 'status_smps': status_smps, 'status_aps': status_aps})

def merge_with_inp(surface_area_series: pd.Series, inp_csv_path: str, tolerance='1h') -> pd.DataFrame:
    """将总表面积与 INP (冰核粒子) 的时间数据配准，并计算 ns 值"""
    df_inp = pd.read_csv(inp_csv_path)
    df_inp['Time'] = pd.to_datetime(df_inp['Time'])
    
    sa_df = surface_area_series.to_frame().reset_index()
    sa_df.columns = ['Time_A', 'Total_Surface_Area(μm2/cm3)']
    sa_df['Time_A'] = pd.to_datetime(sa_df['Time_A'])
    
    # merge_asof 必须对时间排序
    df_inp = df_inp.sort_values('Time')
    sa_df = sa_df.sort_values('Time_A')
    
    result = pd.merge_asof(
        df_inp, sa_df, left_on='Time', right_on='Time_A',
        direction='nearest', tolerance=pd.Timedelta(tolerance)
    )
    
    # 计算 ns 参数 (unit: # / m^2)
    result['n_s'] = result['N_INP(#/L)'] / result['Total_Surface_Area(μm2/cm3)'] * 1e9
    return result

# ==========================================
# 4. 主控调度流程
# ==========================================

def process_aerosol_data(config: dict):
    print("🚀 启动合并程序...")
    
    # 1. 载入原始数据
    smps_df = load_all_files(config["SMPS_DIR"], "SMPS")
    aps_df = load_all_files(config["APS_DIR"], "APS")
    
    # 2. 移除 SMPS 异常时段
    smps_df = remove_bad_periods(smps_df, config["SMPS_BAD_PERIODS"])
    
    # 3. 统一重采样 (设时间点为 Index)
    print(f"🔄 正在按照 {config['RESAMPLE_FREQ']} 频次进行对齐并取均值...")
    smps_df.set_index("__dt__", inplace=True)
    aps_df.set_index("__dt__", inplace=True)
    
    smps_res = smps_df.resample(config["RESAMPLE_FREQ"], closed='left', label='left').mean().dropna(how='all')
    aps_res  = aps_df.resample(config["RESAMPLE_FREQ"], closed='left', label='left').mean().dropna(how='all')
    
    # 4. 清洗与单位换算
    smps_res = clean_and_sort_columns(smps_res)
    aps_res = convert_aps_to_dm(aps_res, rho_eff=config["RHO_EFF"])
    
    # 5. 合并组装宽表
    final_psd_df = merge_smps_aps(smps_res, aps_res, max_aps_dp=config["MAX_APS_DP"])
    print(f"✅ 合并完成！粒径范围: {final_psd_df.columns.min():.2f} nm ~ {final_psd_df.columns.max():.2f} nm")
    
    # 6. 计算衍生物特征
    surface_area = calculate_surface_area(final_psd_df)
    status_df = get_instrument_status(final_psd_df)
    
    # 7. INP 对齐计算
    inp_result = merge_with_inp(surface_area, config["INP_CSV"])
    
    # 8. 持久化存储
    out_dir = Path(config["OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    final_psd_df.to_csv(out_dir / "final_psd(v1.0.2).csv")
    inp_result.to_csv(out_dir / "INP+ns(v1.0.2).csv", index=False)
    status_df.to_csv(out_dir / "instrument_status(v1.0.2).csv", index=False)
    
    print(f"💾 数据已成功输出到目录: {out_dir}")


if __name__ == "__main__":
    # 核心运行配置参数区（请根据本地环境修改此处参数）
    CONFIG = {
        "SMPS_DIR": r"D:\Coding\Data\Lanzhou_aerosol\SMPS_dNdlogDp",
        "APS_DIR":  r"D:\Coding\Data\Lanzhou_aerosol\APS_dNdlogDp",
        "INP_CSV":  r"D:\Coding\Data\Lanzhou_cfdc\processed\N_INP(202409-202509)v2.4.2.csv",
        "OUT_DIR":  r"D:\Coding\Data\Lanzhou_aerosol\SMPS+APS",
        
        # 预处理相关参数
        "RESAMPLE_FREQ": '10min',        # 重采样频次
        "RHO_EFF": 1.5,                  # 气溶胶假设有效密度(g/cm3)
        "MAX_APS_DP": 2500,              # APS 允许最大粒径(nm)
        
        # SMPS 需要剔除浓度异常的阶段 (起止时间对)
        "SMPS_BAD_PERIODS": [
            ("2024-12-14 23:00", "2024-12-16 00:00"),
            ("2024-12-22 19:00", "2024-12-23 12:00"),
            ("2025-03-10 04:00", "2025-03-10 14:00"),
            ("2025-03-25 21:00", "2025-03-25 23:00"),
            ("2025-05-30 15:10", "2025-05-30 15:15"),
            ("2025-06-19 00:00", "2025-06-20 00:00"),
            ("2025-07-23 14:50", "2025-07-23 15:00"),
            ("2025-08-11 17:49", "2025-08-11 17:50"),
            ("2025-08-14 23:00", "2025-08-15 16:00"),
            ("2025-10-16 23:00", "2025-10-19 00:00"),
        ]
    }
    
    # 开始运行
    process_aerosol_data(CONFIG)