from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

def read_tdump(file_path):
    records = []

    with open(file_path, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            parts = line.split()

            # 标准端点记录至少约13列
            if len(parts) < 13:
                continue

            try:
                traj_no = int(parts[0])
                grid_no = int(parts[1])
                year = int(parts[2])
                month = int(parts[3])
                day = int(parts[4])
                hour = int(parts[5])
                minute = int(parts[6])

                age = float(parts[8])
                lat = float(parts[9])
                lon = float(parts[10])
                # 将 0～360° 经度统一转换为 -180～180°
                lon = ((lon + 180.0) % 360.0) - 180.0
                height = float(parts[11])
                pressure = float(parts[12])
            except (ValueError, IndexError):
                continue

            # 排除头文件中的数字行
            if not (1 <= month <= 12):
                continue
            if not (1 <= day <= 31):
                continue
            if not (-90 <= lat <= 90):
                continue
            if not (-180 <= lon <= 180):
                continue

            records.append({
                "trajectory": Path(file_path).name,
                "traj_no": traj_no,
                "grid_no": grid_no,
                "year": year,
                "month": month,
                "day": day,
                "hour": hour,
                "minute": minute,
                "age_h": age,
                "lat": lat,
                "lon": lon,
                "height_m": height,
                "pressure_hpa": pressure,
            })

    return pd.DataFrame(records)

def read_season(folder):
    # 读取指定文件夹中的所有 tdump 文件，并将它们合并为一个 DataFrame。

    all_data = []

    for file_path in sorted(Path(folder).glob("tdump_*")):
        df = read_tdump(file_path)

        if not df.empty:
            all_data.append(df)

    if not all_data:
        raise RuntimeError(f"{folder} 中没有可读取的轨迹")

    return pd.concat(all_data, ignore_index=True)



def endpoint_probability(df, lat_edges, lon_edges):
    
    inside = (
        df["lat"].between(lat_edges[0], lat_edges[-1]) &
        df["lon"].between(lon_edges[0], lon_edges[-1])
    )

    data = df.loc[inside].copy()

    hist, _, _ = np.histogram2d(
        data["lat"],
        data["lon"],
        bins=[lat_edges, lon_edges]
    )

    probability = hist / hist.sum() * 100.0

    return probability

def endpoint_probability_equal_trajectory_weight(
    df,
    lat_edges,
    lon_edges
):
    total_hist = np.zeros(
        (len(lat_edges) - 1, len(lon_edges) - 1),
        dtype=float
    )

    n_trajectories = 0

    for trajectory, group in df.groupby("trajectory"):
        inside = (
            group["lat"].between(
                lat_edges[0],
                lat_edges[-1]
            ) &
            group["lon"].between(
                lon_edges[0],
                lon_edges[-1]
            )
        )

        group = group.loc[inside]

        if group.empty:
            continue

        hist, _, _ = np.histogram2d(
            group["lat"],
            group["lon"],
            bins=[lat_edges, lon_edges]
        )

        # 每条轨迹总权重归一化为1
        hist = hist / hist.sum()

        total_hist += hist
        n_trajectories += 1

    if n_trajectories == 0:
        raise RuntimeError("没有有效轨迹")

    probability = total_hist / n_trajectories * 100.0

    return probability

def trajectory_frequency(df, lat_edges, lon_edges):
    counts = np.zeros(
        (len(lat_edges) - 1, len(lon_edges) - 1),
        dtype=float
    )

    trajectory_names = df["trajectory"].unique()
    n_trajectories = len(trajectory_names)

    for trajectory, group in df.groupby("trajectory"):
        lat_index = np.digitize(
            group["lat"].to_numpy(),
            lat_edges
        ) - 1

        lon_index = np.digitize(
            group["lon"].to_numpy(),
            lon_edges
        ) - 1

        valid = (
            (lat_index >= 0) &
            (lat_index < len(lat_edges) - 1) &
            (lon_index >= 0) &
            (lon_index < len(lon_edges) - 1)
        )

        visited_cells = set(
            zip(lat_index[valid], lon_index[valid])
        )

        for i, j in visited_cells:
            counts[i, j] += 1

    frequency = counts / n_trajectories * 100.0

    return frequency

### 绘图函数 ###

def calculate_map_extent(
    dataframes,
    grid_res=0.5,
    lon_padding=2.0,
    lat_padding=2.0
):
    """
    根据多个季节的轨迹数据自动计算统一地图范围。

    Parameters
    ----------
    dataframes : list[pd.DataFrame]
        需要共同确定范围的数据。
    grid_res : float
        网格分辨率。
    lon_padding, lat_padding : float
        经度和纬度方向的留白，单位为度。

    Returns
    -------
    extent : tuple
        (lon_min, lon_max, lat_min, lat_max)
    """
    all_data = pd.concat(dataframes, ignore_index=True)

    lon_min = all_data["lon"].min() - lon_padding
    lon_max = all_data["lon"].max() + lon_padding
    lat_min = all_data["lat"].min() - lat_padding
    lat_max = all_data["lat"].max() + lat_padding

    # 保证观测站也位于地图范围内
    lon_min = min(lon_min, SITE_LON - lon_padding)
    lon_max = max(lon_max, SITE_LON + lon_padding)
    lat_min = min(lat_min, SITE_LAT - lat_padding)
    lat_max = max(lat_max, SITE_LAT + lat_padding)

    # 将边界对齐到网格分辨率
    lon_min = np.floor(lon_min / grid_res) * grid_res
    lon_max = np.ceil(lon_max / grid_res) * grid_res
    lat_min = np.floor(lat_min / grid_res) * grid_res
    lat_max = np.ceil(lat_max / grid_res) * grid_res

    # 避免超出合法经纬度范围
    lon_min = max(-180.0, lon_min)
    lon_max = min(180.0, lon_max)
    lat_min = max(-90.0, lat_min)
    lat_max = min(90.0, lat_max)

    return lon_min, lon_max, lat_min, lat_max

def plot_probability(
    probability,
    lon_edges,
    lat_edges,
    title,
    output_file,
    vmax=None,
    map_extent=None
):
    fig = plt.figure(figsize=(10, 6))

    ax = plt.axes(
        projection=ccrs.PlateCarree()
    )

    if map_extent is None:
        map_extent = [
            lon_edges[0],
            lon_edges[-1],
            lat_edges[0],
            lat_edges[-1]
        ]

    ax.set_extent(
        map_extent,
        crs=ccrs.PlateCarree()
    )

    # ---------------------------------------------------------
    # 1. 地形背景
    # ---------------------------------------------------------
    # Cartopy 自带的 Natural Earth shaded relief 背景。
    # 它会在没有轨迹概率的区域显示出来。
    ax.stock_img()

    # 可选：在地形背景上增加淡色陆地和海洋。
    # 如果希望保留更明显的地形纹理，可以删除下面 LAND 和 OCEAN 两项。
    ax.add_feature(
        cfeature.LAND,
        facecolor=(1.0, 1.0, 1.0, 0.08),
        zorder=0.5
    )
    ax.add_feature(
        cfeature.OCEAN,
        facecolor=(0.75, 0.88, 0.95, 0.20),
        zorder=0.5
    )

    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=0.6,
        edgecolor="0.25",
        zorder=4
    )
    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.5,
        edgecolor="0.35",
        zorder=4
    )

    # 可选：增加省级行政边界。
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_1_states_provinces_lines",
            scale="50m",
            facecolor="none"
        ),
        linewidth=0.35,
        edgecolor="0.45",
        zorder=4
    )

    # ---------------------------------------------------------
    # 2. 将未覆盖区域，即 probability == 0 的网格掩膜
    # ---------------------------------------------------------
    probability_masked = np.ma.masked_invalid(probability)
    probability_masked = np.ma.masked_less_equal(
        probability_masked,
        0.0
    )

    # 复制色带并将掩膜区域设为完全透明
    cmap = plt.colormaps["turbo"].copy()
    cmap.set_bad(color=(0, 0, 0, 0))

    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        probability_masked,
        cmap=cmap,
        shading="auto",
        vmin=0,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        zorder=2
    )

    # 观测站位置
    ax.scatter(
        SITE_LON,
        SITE_LAT,
        marker="*",
        s=220,
        facecolor="gold",
        edgecolor="black",
        linewidth=1.0,
        transform=ccrs.PlateCarree(),
        zorder=5
    )

    gridlines = ax.gridlines(
        draw_labels=True,
        linewidth=0.4,
        color="gray",
        alpha=0.6,
        linestyle="--"
    )
    gridlines.top_labels = False
    gridlines.right_labels = False

    ax.set_title(title, fontsize=15)

    colorbar = plt.colorbar(
        mesh,
        ax=ax,
        pad=0.03,
        shrink=0.85
    )
    colorbar.set_label("Endpoint probability (%)")

    plt.tight_layout()
    plt.savefig(
        output_file,
        dpi=500,
        bbox_inches="tight"
    )
    plt.close(fig)



SITE_LON = 103.850000
SITE_LAT = 36.033333
GRID_RES = 0.5

spring_df = read_season(
    r"D:\Coding\master0_2025\Lanzhou_backtraj\trajectories\spring"
)

autumn_df = read_season(
    r"D:\Coding\master0_2025\Lanzhou_backtraj\trajectories\autumn"
)

# 根据春季和秋季所有轨迹计算共同地图范围
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = calculate_map_extent(
    [spring_df, autumn_df],
    grid_res=GRID_RES,
    lon_padding=2.0,
    lat_padding=2.0
)
print(
    "共同地图范围："
    f"经度 {LON_MIN:.1f}° 至 {LON_MAX:.1f}°，"
    f"纬度 {LAT_MIN:.1f}° 至 {LAT_MAX:.1f}°"
)
lon_edges = np.arange(
    LON_MIN,
    LON_MAX + GRID_RES * 0.5,
    GRID_RES
)
lat_edges = np.arange(
    LAT_MIN,
    LAT_MAX + GRID_RES * 0.5,
    GRID_RES
)

spring_pdf = endpoint_probability_equal_trajectory_weight(
    spring_df,
    lat_edges,
    lon_edges
)
autumn_pdf = endpoint_probability_equal_trajectory_weight(
    autumn_df,
    lat_edges,
    lon_edges
)

# 两个季节统一色标
common_vmax = max(
    np.nanmax(spring_pdf),
    np.nanmax(autumn_pdf)
)

common_extent = [
    LON_MIN,
    LON_MAX,
    LAT_MIN,
    LAT_MAX
]

plot_probability(
    spring_pdf,
    lon_edges,
    lat_edges,
    title="Spring, 72-h backward trajectories, 3 m AGL",
    output_file=(
        r"D:\Coding\master0_2025\Lanzhou_backtraj"
        r"\figures\spring_pdf.png"
    ),
    vmax=common_vmax,
    map_extent=common_extent
)

plot_probability(
    autumn_pdf,
    lon_edges,
    lat_edges,
    title="Autumn, 72-h backward trajectories, 3 m AGL",
    output_file=(
        r"D:\Coding\master0_2025\Lanzhou_backtraj"
        r"\figures\autumn_pdf.png"
    ),
    vmax=common_vmax,
    map_extent=common_extent
)