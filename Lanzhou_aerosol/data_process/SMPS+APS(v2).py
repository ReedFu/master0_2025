# -*- coding: utf-8 -*-
# 主要功能:
# 1. 读取并合并 SMPS 和 APS 导出数据, 生成气溶胶粒子谱数据集. 同时保留每个 bin 的粒径原始数据, 生成 bin_metadata 数据集.
# 2. 根据 bin 的缺失情况判断 SMPS 和 APS 的运行状态, 并生成运行状态数据集.
# 3. 根据气溶胶粒子谱数据, 计算表面积浓度. 将表面积浓度与 INP 数据进行最近时间匹配, 并计算 ns(#/m2), 生成 INP+ns 数据集.
# 目前版本: v2.0
# 版本记录:
# v2.0: 修改了 SMPS+APS 的合并逻辑; 生成 bin_metadata 数据集.

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import islice


# ============================================================
# 1. 数据读取函数
# ============================================================

def read_instrument_export(path: Path, instr_type: str, encoding="cp1252", scan_lines=100) -> pd.DataFrame:
    """读取单个 SMPS 或 APS 导出文件。"""
    with open(path, "r", encoding=encoding, errors="replace") as f:
        lines = list(islice(f, scan_lines))

    header_idx = None
    sep = "\t"

    for i, line in enumerate(lines):
        if "Sample #" in line and "Date" in line and "Start Time" in line:
            delimiters = {"\t": line.count("\t"), ",": line.count(","), ";": line.count(";")}
            sep = max(delimiters, key=delimiters.get)
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"{path.name}: 前 {scan_lines} 行未找到表头。")

    df = pd.read_csv(path, sep=sep, encoding=encoding, skiprows=header_idx, engine="python")

    # 根据仪器设定参数
    if instr_type == "SMPS":
        time_format = "%Y/%m/%d %H:%M:%S"
        bin_start_idx, bin_end_idx = 9, 112
    elif instr_type == "APS":
        time_format = "%m/%d/%y %H:%M:%S"
        bin_start_idx, bin_end_idx = 5, 56
    else:
        raise ValueError(f"不支持的仪器类型：{instr_type}")

    datetime_str = df["Date"].astype(str).str.strip() + " " + df["Start Time"].astype(str).str.strip()
    timestamps = pd.to_datetime(datetime_str, errors="coerce", format=time_format)

    df = df.loc[timestamps.notna()].copy()
    df["__dt__"] = timestamps.loc[timestamps.notna()].to_numpy()

    bin_cols = list(df.columns[bin_start_idx:bin_end_idx])

    try:
        diameters = np.asarray(bin_cols, dtype=float)
    except ValueError as exc:
        raise ValueError(
            f"{path.name}: 粒径列无法全部转换为浮点数。请检查是否包含 '<0.523' 等欠量程通道。"
            f" 前几个列名为：{bin_cols[:5]}"
        ) from exc

    if np.any(~np.isfinite(diameters)) or np.any(diameters <= 0):
        raise ValueError(f"{path.name}: 粒径列中存在无效值或非正数。")

    result = df[["__dt__"] + bin_cols].copy()

    for col in bin_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    return result


def load_all_files(directory: str, instr_type: str) -> pd.DataFrame:
    """读取目录中的全部 TXT 文件，并按时间合并去重。"""
    files = sorted(Path(directory).glob("*.TXT"))
    print(f"[{instr_type}] 正在加载 {len(files)} 个文件...")

    if not files:
        raise FileNotFoundError(f"{directory} 中没有找到 TXT 文件。")

    all_dfs = []

    for path in files:
        try:
            all_dfs.append(read_instrument_export(path, instr_type))
        except Exception as exc:
            print(f"❌ 读取 {path.name} 失败：{exc}")

    if not all_dfs:
        raise RuntimeError(f"{instr_type}: 所有文件均读取失败。")

    df = pd.concat(all_dfs, ignore_index=True)
    # 按时间排序并去重
    df = df.sort_values("__dt__").drop_duplicates(subset="__dt__", keep="last").reset_index(drop=True)
    return df


def remove_bad_periods(df: pd.DataFrame, bad_periods: list, time_col="__dt__") -> pd.DataFrame:
    """删除指定异常时间段。"""
    if not bad_periods:
        return df

    mask = pd.Series(False, index=df.index)

    for start_str, end_str in bad_periods:
        start = pd.to_datetime(start_str)
        end = pd.to_datetime(end_str)
        mask |= (df[time_col] >= start) & (df[time_col] <= end)

    return df.loc[~mask].reset_index(drop=True)


# ============================================================
# 2. 粒径列处理函数
# ============================================================

def clean_and_sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    """只保留数值粒径列，并按粒径升序排列。"""
    converted_columns = []

    for col in df.columns:
        try:
            converted_columns.append(float(col))
        except (TypeError, ValueError):
            converted_columns.append(col)

    df = df.copy()
    df.columns = converted_columns

    numeric_cols = [col for col in df.columns if isinstance(col, (int, float, np.integer, np.floating))]
    numeric_cols = np.asarray(numeric_cols, dtype=float)

    if len(numeric_cols) < 2:
        raise ValueError("至少需要两个数值粒径通道。")

    if np.any(~np.isfinite(numeric_cols)) or np.any(numeric_cols <= 0):
        raise ValueError("粒径必须为有限正数。")

    if len(np.unique(numeric_cols)) != len(numeric_cols):
        raise ValueError("粒径列中存在重复通道。")

    sorted_cols = np.sort(numeric_cols)
    result = df.loc[:, sorted_cols].copy()
    return result.apply(pd.to_numeric, errors="coerce")


def infer_log_bin_edges(dp):
    """
    根据通道几何中点推导上下边界。

    内部边界为相邻中心的对数中点，首尾边界在对数空间向外延伸半格。
    """
    dp = np.asarray(dp, dtype=float)

    if dp.ndim != 1:
        raise ValueError("粒径必须是一维数组。")

    if len(dp) < 2:
        raise ValueError("至少需要两个粒径通道。")

    if np.any(~np.isfinite(dp)) or np.any(dp <= 0):
        raise ValueError("粒径必须为有限正数。")

    if np.any(np.diff(dp) <= 0):
        raise ValueError("粒径必须严格递增且不能重复。")

    log_mid = np.log10(dp)
    log_edges = np.empty(len(dp) + 1, dtype=float)

    log_edges[1:-1] = (log_mid[:-1] + log_mid[1:]) / 2
    log_edges[0] = log_mid[0] - (log_mid[1] - log_mid[0]) / 2
    log_edges[-1] = log_mid[-1] + (log_mid[-1] - log_mid[-2]) / 2

    edges = 10 ** log_edges
    lower = edges[:-1]
    upper = edges[1:]
    dlogdp = np.diff(log_edges)

    return lower, upper, dlogdp


def create_bin_metadata(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """根据单台仪器的原始通道中心生成 bin 元数据。"""
    centers = df.columns.to_numpy(dtype=float)
    lower, upper, dlogdp = infer_log_bin_edges(centers)

    metadata = pd.DataFrame({
        "center_nm": centers,
        "lower_nm": lower,
        "upper_nm": upper,
        "dlogDp": dlogdp,
        "instrument": instrument,
    })

    return metadata


def convert_Da_to_Dve(aps_df: pd.DataFrame, aps_metadata: pd.DataFrame,
                                      rho_eff=1.5, rho_0=1.0, shape_factor=1.0):
    """
    将 APS 空气动力学粒径 Da 转换为假设密度和形状因子下的体积等效粒径 Dve。参考 DeCarlo et al. (2004) 公式 [28]:

    Dve = Da * sqrt(rho_0 * shape_factor / rho_eff)
    
    其中:
    1. rho_eff: 假设的有效密度 (g/cm3)
    2. rho_0: 标准密度 (1.0 g/cm3)
    3. shape_factor: 形状修正因子 (假设shape factor = 1.0, 表示球形粒子)

    由于全部直径乘同一个常数，dlogDp 保持不变。
    """
    aps_df = aps_df.copy()
    aps_metadata = aps_metadata.copy()

    # 如果检测到第一个数值小于 20, 视作单位是微米(μm), 则转换为纳米(nm)
    if aps_df.columns[0] < 20:
        unit_factor = 1000.0
    else:
        unit_factor = 1.0

    conversion_factor = unit_factor * np.sqrt(rho_0 * shape_factor / rho_eff)

    original_centers = aps_metadata["center_nm"].to_numpy(dtype=float)
    aps_metadata["original_aerodynamic_center"] = original_centers
    aps_metadata["center_nm"] = aps_metadata["center_nm"] * conversion_factor
    aps_metadata["lower_nm"] = aps_metadata["lower_nm"] * conversion_factor
    aps_metadata["upper_nm"] = aps_metadata["upper_nm"] * conversion_factor

    aps_df.columns = aps_metadata["center_nm"].to_numpy(dtype=float)
    aps_df = aps_df.reindex(sorted(aps_df.columns), axis=1)
    aps_metadata = aps_metadata.sort_values("center_nm").reset_index(drop=True)

    recalculated_dlog = np.log10(aps_metadata["upper_nm"]) - np.log10(aps_metadata["lower_nm"])

    if not np.allclose(recalculated_dlog, aps_metadata["dlogDp"], rtol=1e-10, atol=1e-12):
        raise RuntimeError("APS 固定倍数粒径换算前后的 dlogDp 不一致。")

    return aps_df, aps_metadata


# ============================================================
# 3. SMPS 和 APS 拼接函数
# ============================================================

def merge_smps_aps(smps_df: pd.DataFrame, aps_df: pd.DataFrame, smps_metadata: pd.DataFrame,
                   aps_metadata: pd.DataFrame, max_aps_dp=2500):
    """
    合并 SMPS 和 APS。

    选择中心粒径大于 SMPS 最大中心的 APS 通道，重叠区域以 SMPS 为准。
    若最后一个 SMPS bin 与第一个 APS bin 存在重叠，则在重叠区内建立共同拼接边界。
    """
    smps_df = smps_df.copy()
    aps_df = aps_df.copy()
    smps_metadata = smps_metadata.copy()
    aps_metadata = aps_metadata.copy()

    smps_max_center = smps_metadata["center_nm"].max()
    aps_keep = (aps_metadata["center_nm"] > smps_max_center) & (aps_metadata["center_nm"] <= max_aps_dp)

    aps_metadata = aps_metadata.loc[aps_keep].copy().reset_index(drop=True)
    aps_df = aps_df.loc[:, aps_metadata["center_nm"].to_numpy(dtype=float)].copy()

    if aps_metadata.empty:
        raise ValueError("按照当前拼接条件，没有保留下任何 APS 通道。")

    smps_last_idx = smps_metadata["center_nm"].idxmax()
    aps_first_idx = aps_metadata["center_nm"].idxmin()

    smps_lower = smps_metadata.loc[smps_last_idx, "lower_nm"]
    smps_upper = smps_metadata.loc[smps_last_idx, "upper_nm"]
    aps_lower = aps_metadata.loc[aps_first_idx, "lower_nm"]
    aps_upper = aps_metadata.loc[aps_first_idx, "upper_nm"]

    overlap_lower = max(smps_lower, aps_lower)
    overlap_upper = min(smps_upper, aps_upper)

    if overlap_upper > overlap_lower:
        default_stitch = np.sqrt(smps_max_center * aps_metadata.loc[aps_first_idx, "center_nm"])
        stitch_dp = np.clip(default_stitch, overlap_lower, overlap_upper)
        smps_metadata.loc[smps_last_idx, "upper_nm"] = stitch_dp
        aps_metadata.loc[aps_first_idx, "lower_nm"] = stitch_dp
        print(f"🔗 SMPS/APS 拼接边界：{stitch_dp:.3f} nm")
    else:
        print(f"⚠️ SMPS 与 APS 的保留通道之间不存在边界重叠，保留原始边界。")
        print(f"   SMPS 末端上界：{smps_upper:.3f} nm；APS 起始下界：{aps_lower:.3f} nm")

    last_aps_idx = aps_metadata["center_nm"].idxmax()

    if aps_metadata.loc[last_aps_idx, "upper_nm"] > max_aps_dp:
        aps_metadata.loc[last_aps_idx, "upper_nm"] = max_aps_dp

    smps_metadata["dlogDp"] = np.log10(smps_metadata["upper_nm"]) - np.log10(smps_metadata["lower_nm"])
    aps_metadata["dlogDp"] = np.log10(aps_metadata["upper_nm"]) - np.log10(aps_metadata["lower_nm"])

    if (smps_metadata["dlogDp"] <= 0).any() or (aps_metadata["dlogDp"] <= 0).any():
        raise ValueError("拼接后出现非正的 dlogDp。")

    combined_df = pd.concat([smps_df, aps_df], axis=1, join="outer")
    combined_df = combined_df.reindex(sorted(combined_df.columns), axis=1)

    combined_metadata = pd.concat([smps_metadata, aps_metadata], ignore_index=True)
    combined_metadata = combined_metadata.sort_values("center_nm").reset_index(drop=True)

    combined_centers = combined_df.columns.to_numpy(dtype=float)
    metadata_centers = combined_metadata["center_nm"].to_numpy(dtype=float)

    if len(combined_centers) != len(metadata_centers):
        raise RuntimeError("合并后的粒径谱列数与 bin 元数据行数不一致。")

    if not np.allclose(combined_centers, metadata_centers, rtol=1e-10, atol=1e-8):
        raise RuntimeError("合并后的粒径谱列与 bin 元数据中心粒径不一致。")

    return combined_df, combined_metadata


# ============================================================
# 4. 额外输出
# ============================================================

def calculate_surface_area(df: pd.DataFrame, bin_metadata: pd.DataFrame) -> pd.Series:
    """根据保存的真实/推导 bin 宽度计算总表面积浓度，单位为 μm²/cm³。"""
    dp_nm = df.columns.to_numpy(dtype=float)
    metadata_dp = bin_metadata["center_nm"].to_numpy(dtype=float)

    if not np.allclose(dp_nm, metadata_dp, rtol=1e-10, atol=1e-8):
        raise ValueError("粒径谱列与 bin 元数据顺序不一致。")

    dp_um = dp_nm / 1000.0
    dlogdp = bin_metadata["dlogDp"].to_numpy(dtype=float)
    multiplier = np.pi * dp_um ** 2 * dlogdp

    surface_area = df.multiply(multiplier, axis=1).sum(axis=1, min_count=1)
    return surface_area.rename("Total_Surface_Area_um2_cm3")


def get_instrument_status(combined_df: pd.DataFrame, bin_metadata: pd.DataFrame,
                          missing_threshold=0.9) -> pd.DataFrame:
    """根据 bin 的仪器来源判断 SMPS 和 APS 的运行状态。"""
    smps_cols = bin_metadata.loc[bin_metadata["instrument"] == "SMPS", "center_nm"].to_numpy(dtype=float)
    aps_cols = bin_metadata.loc[bin_metadata["instrument"] == "APS", "center_nm"].to_numpy(dtype=float)

    if len(smps_cols) == 0 or len(aps_cols) == 0:
        raise ValueError("bin 元数据中缺少 SMPS 或 APS 通道。")

    smps_missing_fraction = combined_df.loc[:, smps_cols].isna().mean(axis=1)
    aps_missing_fraction = combined_df.loc[:, aps_cols].isna().mean(axis=1)

    # 测试: 统计各种missing_fraction的分布情况
    print(f"SMPS 缺失数据比例分布:\n{smps_missing_fraction.describe()}")
    print(f"APS 缺失数据比例分布:\n{aps_missing_fraction.describe()}")

    status_smps = smps_missing_fraction <= missing_threshold
    status_aps = aps_missing_fraction <= missing_threshold

    return pd.DataFrame({
        "Date": combined_df.index,
        "status_smps": status_smps.to_numpy(),
        "status_aps": status_aps.to_numpy(),
    })


def merge_with_inp(surface_area_series: pd.Series, inp_csv_path: str, tolerance="1h") -> pd.DataFrame:
    """将表面积浓度与 INP 数据进行最近时间匹配，并计算 ns。"""
    df_inp = pd.read_csv(inp_csv_path)
    df_inp["Time"] = pd.to_datetime(df_inp["Time"], errors="coerce")
    df_inp = df_inp.dropna(subset=["Time"]).sort_values("Time")

    sa_df = surface_area_series.rename("Total_Surface_Area(μm2/cm3)").to_frame().reset_index()
    sa_df.columns = ["Time_A", "Total_Surface_Area(μm2/cm3)"]
    sa_df["Time_A"] = pd.to_datetime(sa_df["Time_A"])
    sa_df = sa_df.sort_values("Time_A")

    result = pd.merge_asof(
        df_inp,
        sa_df,
        left_on="Time",
        right_on="Time_A",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )

    valid_surface = result["Total_Surface_Area(μm2/cm3)"] > 0
    result["n_s(#/m2)"] = np.nan
    result.loc[valid_surface, "n_s(#/m2)"] = (
        result.loc[valid_surface, "N_INP(#/L)"]
        / result.loc[valid_surface, "Total_Surface_Area(μm2/cm3)"]
        * 1e9
    )

    return result


# ============================================================
# 5. 主流程
# ============================================================

def process_aerosol_data(config: dict):
    print("启动 SMPS/APS 合并程序...")

    smps_df = load_all_files(config["SMPS_DIR"], "SMPS")
    aps_df = load_all_files(config["APS_DIR"], "APS")

    smps_df = remove_bad_periods(smps_df, config["SMPS_BAD_PERIODS"])

    print(f"正在按照 {config['RESAMPLE_FREQ']} 进行重采样...")

    smps_df = smps_df.set_index("__dt__")
    aps_df = aps_df.set_index("__dt__")

    smps_res = smps_df.resample(config["RESAMPLE_FREQ"], closed="left", label="left").mean().dropna(how="all")
    aps_res = aps_df.resample(config["RESAMPLE_FREQ"], closed="left", label="left").mean().dropna(how="all")

    smps_res = clean_and_sort_columns(smps_res)
    aps_res = clean_and_sort_columns(aps_res)

    smps_metadata = create_bin_metadata(smps_res, "SMPS")
    aps_metadata = create_bin_metadata(aps_res, "APS")

    aps_res, aps_metadata = convert_Da_to_Dve(
        aps_res,
        aps_metadata,
        rho_eff=config["RHO_EFF"],
        rho_0=config["RHO_0"],
        shape_factor=config["SHAPE_FACTOR"],
    )

    final_psd_df, bin_metadata = merge_smps_aps(
        smps_res,
        aps_res,
        smps_metadata,
        aps_metadata,
        max_aps_dp=config["MAX_APS_DP"],
    )

    print(
        f"SMPS 与 APS 合并完成！中心粒径范围：{final_psd_df.columns.min():.2f}–"
        f"{final_psd_df.columns.max():.2f} nm"
    )

    surface_area = calculate_surface_area(final_psd_df, bin_metadata)
    status_df = get_instrument_status(
        final_psd_df,
        bin_metadata,
        missing_threshold=config["STATUS_MISSING_THRESHOLD"],
    )

    inp_result = merge_with_inp(surface_area, config["INP_CSV"], tolerance=config["INP_TOLERANCE"])

    out_dir = Path(config["OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    final_psd_df.index.name = "__dt__"
    final_psd_df.to_csv(out_dir / f"final_psd({config['VERSION']}).csv")
    bin_metadata.to_csv(out_dir / f"psd_bin_metadata({config['VERSION']}).csv", index=False)
    status_df.to_csv(out_dir / f"instrument_status({config['VERSION']}).csv", index=False)
    inp_result.to_csv(out_dir / f"INP+ns({config['VERSION']}).csv", index=False)

    print(f"💾 文件已保存至：{out_dir}")
    print(f"   - final_psd({config['VERSION']}).csv")
    print(f"   - psd_bin_metadata({config['VERSION']}).csv")
    print(f"   - instrument_status({config['VERSION']}).csv")
    print(f"   - INP+ns({config['VERSION']}).csv")


if __name__ == "__main__":
    CONFIG = {
        "SMPS_DIR": r"D:\Coding\Data\Lanzhou_aerosol\SMPS_dNdlogDp",
        "APS_DIR": r"D:\Coding\Data\Lanzhou_aerosol\APS_dNdlogDp",
        "INP_CSV": r"D:\Coding\Data\Lanzhou_cfdc\processed\N_INP(202409-202509)v2.4.2.csv",
        "OUT_DIR": r"D:\Coding\Data\Lanzhou_aerosol\SMPS+APS",

        # 预处理相关参数
        "RESAMPLE_FREQ": "10min",           # 重采样频次
        "RHO_EFF": 1.5,                     # 气溶胶假设有效密度(g/cm3)
        "RHO_0": 1.0,                       # 标准密度(g/cm3)
        "SHAPE_FACTOR": 1.0,                # 气溶胶假设形状因子(球形粒子为1.0)
        "MAX_APS_DP": 2500,                 # APS 最大粒径上限(nm)
        "STATUS_MISSING_THRESHOLD": 0.9,    # 仪器状态判断时允许的缺失数据比例 (我们认为, 仪器只要 10% 以上的通道有数据, 就是在运行的)
        "INP_TOLERANCE": "1h",              # INP 数据与表面积浓度对齐时, 允许的时间差容忍范围
    
        # 异常时间段 (会在合并前删除)
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
        ],

        # 版本号
        "VERSION": "v2.0",
    }

    process_aerosol_data(CONFIG)