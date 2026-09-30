from pathlib import Path
from datetime import timedelta
import os
import shutil
import subprocess
import pandas as pd


# ============================================================
# 1. 用户配置
# ============================================================

MET_DIR = Path(r"D:\Coding\Data\GDAS")

HYSPLIT_DIR = Path(r"C:\HYSPLIT")
HYSPLIT_EXE = HYSPLIT_DIR / "exec" / "hyts_std.exe"

# HYSPLIT实际运行目录
RUN_DIR = HYSPLIT_DIR / "working"

# 项目和结果保存目录
PROJECT_DIR = Path(r"D:\Coding\master0_2025\Lanzhou_backtraj")

LAT = 36.033333
LON = 103.850000

# HYSPLIT默认按m AGL解释
START_HEIGHT_M = 3.0

# 72小时后向轨迹
RUN_HOURS = -72

# 0 = 使用气象资料中的垂直速度
VERTICAL_MOTION = 0

MODEL_TOP_M = 10000.0


# ============================================================
# 2. 根据日期生成GDAS1文件名
# ============================================================

def gdas_filename(date):
    """
    date: pandas.Timestamp或datetime，UTC时间
    返回示例：gdas1.mar25.w1
    """
    month = date.strftime("%b").lower()
    year = date.strftime("%y")

    day = date.day
    week = min((day - 1) // 7 + 1, 5)

    return f"gdas1.{month}{year}.w{week}"


def required_met_files(start_utc, run_hours=-72):
    """
    返回从轨迹开始到结束期间需要的所有GDAS周文件。
    """
    end_utc = start_utc + timedelta(hours=run_hours)

    t0 = min(start_utc, end_utc).normalize()
    t1 = max(start_utc, end_utc).normalize()

    dates = pd.date_range(t0, t1, freq="1D")

    files = []
    for date in dates:
        name = gdas_filename(date)
        if name not in files:
            files.append(name)

    return files


# ============================================================
# 3. 写CONTROL文件
# ============================================================

def write_control(start_utc, output_dir, output_name, met_files):
    output_dir.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    met_dir_string = str(MET_DIR) + os.sep
    output_dir_string = str(output_dir) + os.sep

    lines = [
        start_utc.strftime("%y %m %d %H %M"),
        "1",
        f"{LAT:.6f} {LON:.6f} {START_HEIGHT_M:.1f}",
        str(RUN_HOURS),
        str(VERTICAL_MOTION),
        f"{MODEL_TOP_M:.1f}",
        str(len(met_files)),
    ]

    for met_file in met_files:
        lines.append(met_dir_string)
        lines.append(met_file)

    lines.append(output_dir_string)
    lines.append(output_name)

    control_path = RUN_DIR / "CONTROL"
    control_path.write_text(
        "\n".join(lines) + "\n",
        encoding="ascii"
    )

    return control_path

# ============================================================
# 4. 读取和处理时间
# ============================================================

def read_times(csv_path):
    df = pd.read_csv(csv_path)

    if "Time" not in df.columns:
        raise ValueError(f"{csv_path} 中没有 Time 列")

    times = pd.to_datetime(df["Time"], errors="coerce")

    if times.isna().any():
        bad_rows = df.loc[times.isna()]
        raise ValueError(f"发现无法识别的时间：\n{bad_rows}")

    # 原始时间是北京时间
    times = (
        times.dt.tz_localize("Asia/Shanghai")
        .dt.tz_convert("UTC")
        .dt.round("min")
    )

    # 检查舍入到分钟后是否出现完全重复
    duplicate_count = times.duplicated().sum()
    if duplicate_count > 0:
        print(f"警告：舍入到分钟后有 {duplicate_count} 个重复时间，将去重。")
        times = times.drop_duplicates()

    return times.sort_values().reset_index(drop=True)


# ============================================================
# 5. 运行一个季节
# ============================================================

def run_season(season, csv_path):
    output_dir = PROJECT_DIR / "trajectories" / season
    output_dir.mkdir(parents=True, exist_ok=True)

    times = read_times(csv_path)

    print(f"\n{season}: 共读取 {len(times)} 个UTC起始时间")

    failed = []

    for i, start_utc in enumerate(times, start=1):
        start_naive = start_utc.tz_localize(None)

        output_name = (
            f"tdump_{season}_"
            f"{start_naive.strftime('%Y%m%d_%H%M')}"
        )
        output_path = output_dir / output_name

        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"[{i}/{len(times)}] 已存在，跳过：{output_name}")
            continue

        met_files = required_met_files(start_naive, RUN_HOURS)

        # 检查气象文件
        missing = [
            name for name in met_files
            if not (MET_DIR / name).exists()
        ]

        if missing:
            print(
                f"[{i}/{len(times)}] 缺少气象文件："
                f"{', '.join(missing)}"
            )
            failed.append((start_utc, "missing met files", missing))
            continue

        write_control(
            start_utc=start_naive,
            output_dir=output_dir,
            output_name=output_name,
            met_files=met_files
        )

        print(
            f"[{i}/{len(times)}] "
            f"{start_utc.strftime('%Y-%m-%d %H:%M UTC')} "
            f"使用 {met_files}"
        )

        result = subprocess.run(
            [str(HYSPLIT_EXE)],
            cwd=str(RUN_DIR),
            capture_output=True,
            text=True
        )

        if not output_path.exists() or output_path.stat().st_size == 0:
            print(f"  失败：没有生成 {output_name}")
            failed.append(
                (
                    start_utc,
                    "no output",
                    result.stdout + "\n" + result.stderr
                )
            )

            message_file = RUN_DIR / "MESSAGE"
            if message_file.exists():
                message_copy = (
                    output_dir /
                    f"MESSAGE_FAILED_{start_naive:%Y%m%d_%H%M}.txt"
                )
                shutil.copy(message_file, message_copy)

    print(f"\n{season}运行结束，失败数量：{len(failed)}")

    if failed:
        failed_path = output_dir / "failed_runs.txt"
        with failed_path.open("w", encoding="utf-8") as f:
            for item in failed:
                f.write(repr(item) + "\n")
        print(f"失败记录已写入：{failed_path}")


# ============================================================
# 6. 主程序
# ============================================================

if __name__ == "__main__":
    if not HYSPLIT_EXE.exists():
        raise FileNotFoundError(
            f"找不到HYSPLIT程序：{HYSPLIT_EXE}"
        )

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    run_season("spring",PROJECT_DIR / "input" / "spring_time.csv")

    #run_season("autumn",PROJECT_DIR / "input" / "autumn_time.csv")