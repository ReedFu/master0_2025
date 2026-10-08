import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

# ============================================================
# 1. 全局绘图设置
# ============================================================
def configure_plot_style():
    """配置适合论文插图的全局绘图样式"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.unicode_minus': False,

        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman',
        'mathtext.bf': 'Times New Roman:bold',

        'axes.linewidth': 1.1,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,

        'figure.dpi': 150,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight'
    })


configure_plot_style()


# ============================================================
# 2. PMF 因子名称：保持原始设定
# ============================================================
FACTOR_NAMES = {
    'spring': {
        'Factor 1': 'Coal combustion',
        'Factor 2': 'Secondary formation',
        'Factor 3': 'Vehicle emissions',
        'Factor 4': 'Mixed industrial emissions',
        'Factor 5': 'Mineral dust',
    },
    'summer': {
        'Factor 1': 'Secondary nitrate',
        'Factor 2': 'Vehicle emissions',
        'Factor 3': 'Secondary sulfate',
        'Factor 4': 'Mineral dust',
        'Factor 5': 'Mixed industrial emissions',
    },
    'autumn': {
        'Factor 1': 'Mineral dust',
        'Factor 2': 'Power plant',
        'Factor 3': 'Secondary formation',
        'Factor 4': 'Vehicle emissions',
    },
    'winter': {
        'Factor 1': 'Secondary formation',
        'Factor 2': 'Mineral dust',
        'Factor 3': 'Smelting industry',
        'Factor 4': 'Vehicle emissions',
        'Factor 5': 'Coal combustion',
        'Factor 6': 'Power plant', 
        'Factor 7': 'Fireworks',
    }
}


# ============================================================
# 3. 季节 PM2.5 平均浓度与标准差
#    单位：μg m−3
# ============================================================
PM25_STATS = {
    'winter': {'mean': 65.22, 'std': 27.21},
    'spring': {'mean': 47.65, 'std': 45.18},
    'summer': {'mean': 25.14, 'std': 7.80},
    'autumn': {'mean': 33.52, 'std': 18.86}
}


# ============================================================
# 4. 统一来源颜色：保持原始配色
# ============================================================
COLOR_MAP = {
    'Coal combustion': '#000000',             # 黑色
    'Mineral dust': '#FFA500',                # 橙色
    'Vehicle emissions': '#808080',           # 灰色
    'Secondary formation': '#0000FF',         # 蓝色
    'Smelting industry': '#556B2F',           # 暗橄榄绿
    'Fireworks': '#008000',                   # 绿色
    'Power plant': '#FF00FF',                 # 品红色
    'Mixed industrial emissions': '#FF4500'   # 橙红色
}


# 来源在累积柱中的统一堆叠顺序。
# 所有季节均使用同一顺序，便于横向比较。
SOURCE_ORDER = [
    'Vehicle emissions',
    'Secondary formation',
    'Mineral dust',
    'Smelting industry',
    'Power plant',
    'Mixed industrial emissions',
    'Fireworks',
    'Coal combustion'
]


# ============================================================
# 5. PMF 结果文件路径
# ============================================================
FILES = {
    'winter': r"D:\Coding\Data\Lanzhou_chemical\PMF Output (SigOnly)\DJF_profiles.csv",
    'spring': r"D:\Coding\Data\Lanzhou_chemical\PMF Output (SigOnly)\MAM_profiles.csv",
    'summer': r"D:\Coding\Data\Lanzhou_chemical\PMF Output (SigOnly)\JJA_profiles.csv",
    'autumn': r"D:\Coding\Data\Lanzhou_chemical\PMF Output (SigOnly)\SON_profiles.csv"
}


# 图件输出路径
OUTPUT_FILE = (
    r"D:\Coding\master0_2025\Thesis"
    r"\PM2.5_Seasonal_Source_Contribution_and_Concentration(sig).png"
)


# ============================================================
# 6. 数据读取与整理函数
# ============================================================
def extract_pm25_percentages(filepath):
    """
    从 PMF 输出文件中提取 PM2.5 行对应的各因子百分比贡献。

    文件中读取的位置：
    'Factor Profiles (% of species sum)'
    与
    'Factor Profiles (% of total variable)'
    两个标记之间。

    返回：
        list[float]，按 Factor 1、Factor 2、... 的顺序排列。
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"未找到 PMF 数据文件：\n{filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_target_section = False

    for line in lines:
        if "Factor Profiles (% of species sum)" in line:
            in_target_section = True
            continue

        if "Factor Profiles (% of total variable)" in line:
            break

        if in_target_section and "PM2.5" in line:
            parts = line.strip().split(',')

            # 保持与原代码一致：从第3列开始读取数值
            values = []
            for value in parts[2:]:
                value = value.strip()
                if value == '':
                    continue
                try:
                    values.append(float(value))
                except ValueError:
                    # 若存在非数值字符，则跳过
                    continue

            if values:
                return values

    raise ValueError(
        f"无法从以下文件中读取 PM2.5 的因子贡献数据：\n{filepath}\n"
        "请检查文件中是否包含：'Factor Profiles (% of species sum)' "
        "以及 PM2.5 对应的数据行。"
    )


def merge_secondary_sources(original_names, original_values):
    """
    合并二次硝酸盐和二次硫酸盐：
    Secondary nitrate + Secondary sulfate → Secondary formation
    """
    merged_data = {}

    for name, value in zip(original_names, original_values):
        if name in ['Secondary nitrate', 'Secondary sulfate']:
            name = 'Secondary formation'

        merged_data[name] = merged_data.get(name, 0) + value

    return merged_data


def get_season_source_data(season):
    """
    读取指定季节数据，并转换为统一来源名称的数据字典。

    返回：
        dict，例如：
        {
            'Mineral dust': 45.0,
            'Vehicle emissions': 25.0,
            ...
        }
    """
    values = extract_pm25_percentages(FILES[season])
    factor_dict = FACTOR_NAMES[season]

    expected_factor_number = len(factor_dict)

    if len(values) != expected_factor_number:
        raise ValueError(
            f"{season.capitalize()} 数据中的 PM2.5 数值数量为 {len(values)}，"
            f"但 FACTOR_NAMES 中定义了 {expected_factor_number} 个因子。\n"
            f"请检查文件格式或 {season} 的因子名称映射。"
        )

    original_names = [
        factor_dict[f"Factor {i + 1}"]
        for i in range(len(values))
    ]

    merged_data = merge_secondary_sources(original_names, values)

    # 因浮点误差或原始文件四舍五入，统一标准化到 100%
    total = sum(merged_data.values())

    if total <= 0:
        raise ValueError(f"{season.capitalize()} 的 PM2.5 来源贡献总和小于或等于 0。")

    normalized_data = {
        source: value / total * 100
        for source, value in merged_data.items()
    }

    return normalized_data


# ============================================================
# 7. 读取四季来源贡献数据
# ============================================================
SEASONS = ['winter', 'spring', 'summer', 'autumn']
SEASON_LABELS = ['Winter', 'Spring', 'Summer', 'Autumn']

seasonal_source_data = {
    season: get_season_source_data(season)
    for season in SEASONS
}

# 在控制台输出各季节来源贡献，便于核查
print("=" * 65)
print("Seasonal PM2.5 source contributions (%)")
print("=" * 65)

for season in SEASONS:
    print(f"\n{season.capitalize()}:")
    for source in SOURCE_ORDER:
        value = seasonal_source_data[season].get(source, 0)
        if value > 0:
            print(f"  {source:<30s}: {value:6.2f}%")
    print(f"  {'Total':<30s}: {sum(seasonal_source_data[season].values()):6.2f}%")


# ============================================================
# 8. 绘制图件
# ============================================================
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(10, 9),
    gridspec_kw={'height_ratios': [1.25, 1]},
    constrained_layout=False
)

x = np.arange(len(SEASONS))


# ------------------------------------------------------------
# (a) 100% 累积柱状图：PMF 来源百分比贡献
# ------------------------------------------------------------
bottom = np.zeros(len(SEASONS))
legend_handles = []

for source in SOURCE_ORDER:
    contributions = np.array([
        seasonal_source_data[season].get(source, 0)
        for season in SEASONS
    ])

    color = COLOR_MAP[source]

    bars = ax1.bar(
        x,
        contributions,
        bottom=bottom,
        #width=0.68,
        width=0.3,
        color=color,
        edgecolor='black',
        linewidth=0.9,
        label=source,
        zorder=3
    )

    # 在较大的色块中标注百分比；
    # 小于 5% 的组分不在图内标注，以避免文字拥挤。
    for bar, value, current_bottom in zip(bars, contributions, bottom):
        if value >= 5:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                current_bottom + value / 2,
                f'{value:.0f}%',
                ha='center',
                va='center',
                fontsize=10,
                color='white',
                fontweight='normal'
            )

    bottom += contributions
    legend_handles.append(Patch(facecolor=color, edgecolor='none', label=source))

ax1.set_ylim(0, 100)
ax1.set_xlim(-0.6, len(SEASONS) - 0.4)
ax1.set_xticks(x)
ax1.set_xticklabels(SEASON_LABELS)
ax1.set_ylabel('Source contribution (%)')
ax1.set_title('(a) Seasonal PM$_{2.5}$ source contributions', loc='left', fontweight='bold')

ax1.set_yticks(np.arange(0, 101, 20))
ax1.yaxis.grid(True, linestyle='--', linewidth=0.7, color='0.80', zorder=0)
ax1.set_axisbelow(True)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)


# ------------------------------------------------------------
# (b) PM2.5 平均浓度及标准差
# ------------------------------------------------------------
means = np.array([PM25_STATS[season]['mean'] for season in SEASONS])
stds = np.array([PM25_STATS[season]['std'] for season in SEASONS])

# 使用中性灰色，以避免与来源类别颜色产生混淆
concentration_bar_colors = ['#4D4D4D', '#6E6E6E', '#8A8A8A', '#A6A6A6']

bars2 = ax2.bar(
    x,
    means,
    #width=0.58,
    width=0.3,
    color=concentration_bar_colors,
    edgecolor='black',
    linewidth=0.8,
    zorder=3
)

ax2.errorbar(
    x,
    means,
    yerr=stds,
    fmt='none',
    ecolor='black',
    elinewidth=1.3,
    capsize=5,
    capthick=1.3,
    zorder=4
)

# 标注均值 ± 标准差
for xi, mean, std in zip(x, means, stds):
    ax2.text(
        xi,
        mean + std + 3,
        f'{mean:.1f} ± {std:.1f}',
        ha='center',
        va='bottom',
        fontsize=10
    )

ax2.set_xlim(-0.6, len(SEASONS) - 0.4)
ax2.set_xticks(x)
ax2.set_xticklabels(SEASON_LABELS)
ax2.set_ylabel(r'PM$_{2.5}$ concentration ($\mu$g m$^{-3}$)')
ax2.set_title(
    r'(b) Seasonal mean PM$_{2.5}$ concentration (mean $\pm$ SD)',
    loc='left',
    fontweight='bold'
)

# 自动为误差线预留上方空间
upper_limit = max(means + stds) * 1.20
ax2.set_ylim(0, upper_limit)

ax2.yaxis.grid(True, linestyle='--', linewidth=0.7, color='0.80', zorder=0)
ax2.set_axisbelow(True)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)


# ============================================================
# 9. 全局图例与排版
# ============================================================
fig.legend(
    handles=legend_handles,
    labels=SOURCE_ORDER,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.98),
    ncol=4,
    frameon=True,
    edgecolor='0.75',
    columnspacing=1.5,
    handlelength=1.8
)

# 为顶部图例和子图间距预留空间
fig.subplots_adjust(
    top=0.86,
    bottom=0.08,
    left=0.12,
    right=0.97,
    hspace=0.3
)

# 创建输出目录（若不存在）
output_dir = os.path.dirname(OUTPUT_FILE)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

plt.savefig(OUTPUT_FILE, dpi=600, bbox_inches='tight')
plt.show()

print("\nFigure saved to:")
print(OUTPUT_FILE)