import numpy as np
import pandas as pd


PSD_FILE = (
    r"D:\Coding\Data\Lanzhou_aerosol\SMPS+APS"
    r"\final_psd(v2.0).csv"
)

BIN_METADATA_FILE = (
    r"D:\Coding\Data\Lanzhou_aerosol\SMPS+APS"
    r"\psd_bin_metadata(v2.0).csv"
)

STATUS_FILE = (
    r"D:\Coding\Data\Lanzhou_aerosol\SMPS+APS"
    r"\instrument_status(v2.0).csv"
)

CONC_OUTPUT = r"D:\Coding\Data\Lanzhou_aerosol\SMPS+APS\aer_conc_results.csv"
ELEMENT_FILE = r"D:\Coding\Data\Lanzhou_chemical\Corr(INP_vs_element).csv"
MERGE_OUTPUT = r"D:\Coding\Data\Lanzhou_chemical\To_Corr_heatmap.csv"


# ============================================================
# 1. 通用函数
# ============================================================

def convert_to_boolean(series: pd.Series) -> pd.Series:
    """将常见布尔值表达统一转换为 pandas 可空布尔类型。"""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "on": True,
        "off": False,
    }

    original_notna = series.notna()
    converted = series.astype("string").str.strip().str.lower().map(mapping).astype("boolean")
    invalid = original_notna & converted.isna()

    if invalid.any():
        invalid_values = series.loc[invalid].unique().tolist()
        raise ValueError(f"状态列 {series.name!r} 中存在无法识别的值：{invalid_values}")

    return converted


def validate_and_align_metadata(final_psd_df: pd.DataFrame,
                                bin_metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """检查粒径谱列和 bin 元数据，并统一按照中心粒径排序。"""
    required_columns = {"center_nm", "lower_nm", "upper_nm", "dlogDp", "instrument"}
    missing_columns = required_columns.difference(bin_metadata.columns)

    if missing_columns:
        raise KeyError(f"bin 元数据缺少列：{sorted(missing_columns)}")

    diameters = pd.to_numeric(final_psd_df.columns, errors="coerce").to_numpy(dtype=float)

    if np.any(~np.isfinite(diameters)):
        invalid_columns = final_psd_df.columns[~np.isfinite(diameters)].tolist()
        raise ValueError(f"以下粒径谱列名无法转换为数值：{invalid_columns}")

    if np.any(diameters <= 0):
        raise ValueError("粒径必须全部大于 0。")

    if len(np.unique(diameters)) != len(diameters):
        raise ValueError("粒径谱中存在重复中心粒径。")

    final_psd_df = final_psd_df.copy()
    final_psd_df.columns = diameters

    bin_metadata = bin_metadata.copy()

    for col in ["center_nm", "lower_nm", "upper_nm", "dlogDp"]:
        bin_metadata[col] = pd.to_numeric(bin_metadata[col], errors="coerce")

    if bin_metadata[["center_nm", "lower_nm", "upper_nm", "dlogDp"]].isna().any().any():
        raise ValueError("bin 元数据中存在无法转换为数值的值。")

    if len(bin_metadata) != final_psd_df.shape[1]:
        raise ValueError(
            f"粒径谱有 {final_psd_df.shape[1]} 个通道，但 bin 元数据有 {len(bin_metadata)} 行。"
        )

    final_psd_df = final_psd_df.reindex(sorted(final_psd_df.columns), axis=1)
    bin_metadata = bin_metadata.sort_values("center_nm").reset_index(drop=True)

    psd_centers = final_psd_df.columns.to_numpy(dtype=float)
    metadata_centers = bin_metadata["center_nm"].to_numpy(dtype=float)

    if not np.allclose(psd_centers, metadata_centers, rtol=1e-10, atol=1e-8):
        mismatch = pd.DataFrame({"psd_center": psd_centers, "metadata_center": metadata_centers})
        raise ValueError(f"粒径谱列与 bin 元数据中心粒径不一致：\n{mismatch.head(10)}")

    lower = bin_metadata["lower_nm"].to_numpy(dtype=float)
    upper = bin_metadata["upper_nm"].to_numpy(dtype=float)
    stored_dlog = bin_metadata["dlogDp"].to_numpy(dtype=float)

    if np.any(lower <= 0) or np.any(upper <= lower):
        raise ValueError("bin 元数据中存在无效上下边界。")

    calculated_dlog = np.log10(upper) - np.log10(lower)

    if not np.allclose(stored_dlog, calculated_dlog, rtol=1e-10, atol=1e-12):
        raise ValueError("bin 元数据中的 dlogDp 与上下边界不一致。")

    final_psd_df = final_psd_df.apply(pd.to_numeric, errors="coerce")
    return final_psd_df, bin_metadata


def calculate_overlap_dlog(bin_metadata: pd.DataFrame, range_low: float, range_high: float) -> np.ndarray:
    """
    计算每个 bin 与目标粒径范围重叠部分的 log10 宽度。

    返回值为 0 表示该 bin 与目标范围不重叠。
    """
    if range_low <= 0 or range_high <= range_low:
        raise ValueError("目标粒径范围必须满足 0 < low < high。")

    lower = bin_metadata["lower_nm"].to_numpy(dtype=float)
    upper = bin_metadata["upper_nm"].to_numpy(dtype=float)

    overlap_lower = np.maximum(lower, range_low)
    overlap_upper = np.minimum(upper, range_high)

    overlap_dlog = np.zeros(len(bin_metadata), dtype=float)
    valid = overlap_upper > overlap_lower
    overlap_dlog[valid] = np.log10(overlap_upper[valid]) - np.log10(overlap_lower[valid])

    return overlap_dlog


# ============================================================
# 2. 读取数据
# ============================================================

final_psd_df = pd.read_csv(PSD_FILE, index_col="__dt__", parse_dates=["__dt__"])
status_df = pd.read_csv(STATUS_FILE, index_col="Date", parse_dates=["Date"])
bin_metadata = pd.read_csv(BIN_METADATA_FILE)

final_psd_df = final_psd_df.sort_index()
status_df = status_df.sort_index()

if final_psd_df.index.has_duplicates:
    raise ValueError("粒径谱文件中存在重复时间戳。")

if status_df.index.has_duplicates:
    raise ValueError("仪器状态文件中存在重复时间戳。")


# ============================================================
# 3. 处理仪器状态
# ============================================================

for status_column in ["status_smps", "status_aps"]:
    if status_column not in status_df.columns:
        raise KeyError(f"状态文件中缺少列：{status_column}")

    status_df[status_column] = convert_to_boolean(status_df[status_column])

status_aligned = status_df.reindex(final_psd_df.index)


# ============================================================
# 4. 检查粒径谱和 bin 元数据
# ============================================================

final_psd_df, bin_metadata = validate_and_align_metadata(final_psd_df, bin_metadata)


# ============================================================
# 5. 定义积分范围
# ============================================================

range_settings = [
    {"low": 10, "high": 500, "required_status": ["status_smps"]},
    {"low": 10, "high": 1000, "required_status": ["status_smps", "status_aps"]},
    {"low": 500, "high": 2500, "required_status": ["status_smps", "status_aps"]},
    {"low": 1000, "high": 2500, "required_status": ["status_aps"]},
]


# ============================================================
# 6. 计算各粒径范围数浓度
# ============================================================

results = {}

for setting in range_settings:
    range_low = setting["low"]
    range_high = setting["high"]
    required_status = setting["required_status"]

    overlap_dlog = calculate_overlap_dlog(bin_metadata, range_low, range_high)
    channel_mask = overlap_dlog > 0
    number_of_channels = int(channel_mask.sum())

    if number_of_channels == 0:
        raise ValueError(f"粒径谱中没有与 {range_low}–{range_high} nm 重叠的通道。")

    selected_psd = final_psd_df.iloc[:, channel_mask]
    selected_widths = overlap_dlog[channel_mask]

    delta_n = selected_psd.multiply(selected_widths, axis=1)

    concentration = delta_n.sum(axis=1, min_count=1)

    valid_status = pd.Series(True, index=final_psd_df.index, dtype="boolean")

    for status_column in required_status:
        valid_status = valid_status & status_aligned[status_column].eq(True)

    valid_status = valid_status.fillna(False)
    concentration = concentration.where(valid_status)

    column_name = f"N_{range_low}-{range_high}nm"
    results[column_name] = concentration


# ============================================================
# 7. 生成数浓度结果
# ============================================================

number_conc_df = pd.DataFrame(results, index=final_psd_df.index)
number_conc_df.index.name = "__dt__"

print(number_conc_df.head())

number_conc_df.to_csv(CONC_OUTPUT, index=True)
print(f"结果已保存至：{CONC_OUTPUT}")

# ============================================================
# 8. 与元素数据进行时间匹配
# ============================================================

df_element = pd.read_csv(ELEMENT_FILE)

if "datetime" in df_element.columns:
    df_element = df_element.drop(columns=["datetime"])

if "Time" not in df_element.columns:
    raise KeyError("元素数据文件中缺少 Time 列。")

df_element["Time"] = pd.to_datetime(df_element["Time"], errors="coerce")
df_element = df_element.dropna(subset=["Time"]).sort_values("Time")

number_for_merge = number_conc_df.reset_index().sort_values("__dt__")

result = pd.merge_asof(
    df_element,
    number_for_merge,
    left_on="Time",
    right_on="__dt__",
    direction="nearest",
    tolerance=pd.Timedelta("1h"),
)

result.to_csv(MERGE_OUTPUT, index=False)
print(f"结果已保存至：{MERGE_OUTPUT}")