from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

SITE_LON = 103.850000
SITE_LAT = 36.033333
GRID_RES = 0.5

SPRING_DIR = Path(
    r"D:\Coding\master0_2025\Lanzhou_backtraj"
    r"\trajectories\spring"
)
AUTUMN_DIR = Path(
    r"D:\Coding\master0_2025\Lanzhou_backtraj"
    r"\trajectories\autumn"
)
OUTPUT_FILE = Path(
    r"D:\Coding\master0_2025\Lanzhou_backtraj"
    r"\figures\spring_autumn_endpoint_frequency.png"
)


def read_tdump(file_path):
    records = []

    with open(file_path, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            parts = line.split()

            # 标准端点记录至少约 13 列
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
    """读取指定文件夹中的所有 tdump 文件并合并。"""
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
        df["lat"].between(lat_edges[0], lat_edges[-1])
        & df["lon"].between(lon_edges[0], lon_edges[-1])
    )
    data = df.loc[inside]

    hist, _, _ = np.histogram2d(
        data["lat"],
        data["lon"],
        bins=[lat_edges, lon_edges],
    )

    if hist.sum() == 0:
        raise RuntimeError("地图范围内没有有效端点")

    return hist / hist.sum() * 100.0


def endpoint_probability_equal_trajectory_weight(
    df,
    lat_edges,
    lon_edges,
):
    """每条轨迹的端点总权重相同，返回各网格的端点百分比。"""
    total_hist = np.zeros(
        (len(lat_edges) - 1, len(lon_edges) - 1),
        dtype=float,
    )
    n_trajectories = 0

    for _, group in df.groupby("trajectory"):
        inside = (
            group["lat"].between(lat_edges[0], lat_edges[-1])
            & group["lon"].between(lon_edges[0], lon_edges[-1])
        )
        group = group.loc[inside]

        if group.empty:
            continue

        hist, _, _ = np.histogram2d(
            group["lat"],
            group["lon"],
            bins=[lat_edges, lon_edges],
        )

        # 每条轨迹的端点总权重归一化为 1
        total_hist += hist / hist.sum()
        n_trajectories += 1

    if n_trajectories == 0:
        raise RuntimeError("地图范围内没有有效轨迹端点")

    return total_hist / n_trajectories * 100.0


def trajectory_frequency(df, lat_edges, lon_edges):
    """计算每个网格被多少比例的轨迹经过。"""
    counts = np.zeros(
        (len(lat_edges) - 1, len(lon_edges) - 1),
        dtype=float,
    )

    n_trajectories = df["trajectory"].nunique()
    if n_trajectories == 0:
        raise RuntimeError("没有有效轨迹")

    for _, group in df.groupby("trajectory"):
        lat_index = np.digitize(
            group["lat"].to_numpy(), lat_edges
        ) - 1
        lon_index = np.digitize(
            group["lon"].to_numpy(), lon_edges
        ) - 1

        valid = (
            (lat_index >= 0)
            & (lat_index < len(lat_edges) - 1)
            & (lon_index >= 0)
            & (lon_index < len(lon_edges) - 1)
        )

        visited_cells = set(zip(lat_index[valid], lon_index[valid]))
        for i, j in visited_cells:
            counts[i, j] += 1

    return counts / n_trajectories * 100.0


def calculate_map_extent(
    dataframes,
    grid_res=0.5,
    lon_padding=2.0,
    lat_padding=2.0,
):
    """根据多个季节的轨迹数据计算统一地图范围。"""
    all_data = pd.concat(dataframes, ignore_index=True)

    lon_min = min(
        all_data["lon"].min() - lon_padding,
        SITE_LON - lon_padding,
    )
    lon_max = max(
        all_data["lon"].max() + lon_padding,
        SITE_LON + lon_padding,
    )
    lat_min = min(
        all_data["lat"].min() - lat_padding,
        SITE_LAT - lat_padding,
    )
    lat_max = max(
        all_data["lat"].max() + lat_padding,
        SITE_LAT + lat_padding,
    )

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


def plot_probability_on_axis(
    ax,
    probability,
    lon_edges,
    lat_edges,
    title,
    map_extent,
    vmax,
    trajectory_count,
    show_left_labels=True,
):
    """在指定子图上绘制一个季节的端点分布。"""
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())

    # 地形背景
    ax.stock_img()
    ax.add_feature(
        cfeature.LAND,
        facecolor=(1.0, 1.0, 1.0, 0.08),
        zorder=0.5,
    )
    ax.add_feature(
        cfeature.OCEAN,
        facecolor=(0.75, 0.88, 0.95, 0.20),
        zorder=0.5,
    )
    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=0.6,
        edgecolor="0.25",
        zorder=4,
    )
    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.5,
        edgecolor="0.35",
        zorder=4,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_1_states_provinces_lines",
            scale="50m",
            facecolor="none",
        ),
        linewidth=0.35,
        edgecolor="0.45",
        zorder=4,
    )

    # 无端点的网格保持透明，以显示底图
    probability_masked = np.ma.masked_invalid(probability)
    probability_masked = np.ma.masked_less_equal(
        probability_masked, 0.0
    )

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
        zorder=2,
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
        zorder=5,
    )

    gridlines = ax.gridlines(
        draw_labels=True,
        linewidth=0.4,
        color="gray",
        alpha=0.6,
        linestyle="--",
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.left_labels = show_left_labels

    ax.set_title(title, fontsize=14, pad=12)

    # 在子图右上角标注轨迹数量
    ax.text(
        0.98,
        0.98,
        f"number of trajectories = {trajectory_count}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.5",
            "alpha": 0.85,
        },
        zorder=10,
    )

    return mesh


def plot_season_comparison(
    spring_probability,
    autumn_probability,
    spring_trajectory_count,
    autumn_trajectory_count,
    lon_edges,
    lat_edges,
    map_extent,
    output_file,
):
    """将春、秋两季绘制为 (a)、(b) 子图，并共用一个色标。"""
    common_vmax = max(
        float(np.nanmax(spring_probability)),
        float(np.nanmax(autumn_probability)),
    )
    if common_vmax <= 0:
        raise RuntimeError("两个季节均没有可绘制的端点分布")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(17, 7),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    plot_probability_on_axis(
        axes[0],
        spring_probability,
        lon_edges,
        lat_edges,
        title="(a) Spring, 72-h backward trajectories, 3 m AGL",
        map_extent=map_extent,
        vmax=common_vmax,
        trajectory_count=spring_trajectory_count,
    )
    mesh = plot_probability_on_axis(
        axes[1],
        autumn_probability,
        lon_edges,
        lat_edges,
        title="(b) Autumn, 72-h backward trajectories, 3 m AGL",
        map_extent=map_extent,
        vmax=common_vmax,
        trajectory_count=autumn_trajectory_count,
        show_left_labels=False,
    )

    # 两个子图共用色标
    colorbar = fig.colorbar(
        mesh,
        ax=axes,
        orientation="horizontal",
        fraction=0.045,
        pad=0.09,
        shrink=0.7,
    )
    colorbar.set_label("Endpoint probability (%)")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=500, bbox_inches="tight")
    plt.close(fig)


def main():
    spring_df = read_season(SPRING_DIR)
    autumn_df = read_season(AUTUMN_DIR)

    # 轨迹数量按文件名统计，与概率计算中的分组方式一致
    spring_trajectory_count = spring_df["trajectory"].nunique()
    autumn_trajectory_count = autumn_df["trajectory"].nunique()

    # 春、秋两季使用相同的地图范围和网格
    lon_min, lon_max, lat_min, lat_max = calculate_map_extent(
        [spring_df, autumn_df],
        grid_res=GRID_RES,
        lon_padding=2.0,
        lat_padding=2.0,
    )
    print(
        "共同地图范围："
        f"经度 {lon_min:.1f}° 至 {lon_max:.1f}°，"
        f"纬度 {lat_min:.1f}° 至 {lat_max:.1f}°"
    )

    lon_edges = np.arange(
        lon_min, lon_max + GRID_RES * 0.5, GRID_RES
    )
    lat_edges = np.arange(
        lat_min, lat_max + GRID_RES * 0.5, GRID_RES
    )

    spring_pdf = endpoint_probability_equal_trajectory_weight(
        spring_df, lat_edges, lon_edges
    )
    autumn_pdf = endpoint_probability_equal_trajectory_weight(
        autumn_df, lat_edges, lon_edges
    )

    plot_season_comparison(
        spring_probability=spring_pdf,
        autumn_probability=autumn_pdf,
        spring_trajectory_count=spring_trajectory_count,
        autumn_trajectory_count=autumn_trajectory_count,
        lon_edges=lon_edges,
        lat_edges=lat_edges,
        map_extent=[lon_min, lon_max, lat_min, lat_max],
        output_file=OUTPUT_FILE,
    )
    print(f"图片已保存：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()