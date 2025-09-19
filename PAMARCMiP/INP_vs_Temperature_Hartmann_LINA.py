import matplotlib.pyplot as plt
import pandas as pd
import pathlib

data_dir = pathlib.Path(r"D:\Data\PAMARCMiP\Hartmann-etal_2019_INP-PAMARCMIP\datasets")# 数据文件夹路径
data_files = list(data_dir.glob('*.tab'))

miny, maxy = 1, 0

for data_file in data_files:

    df = pd.read_table(data_file, skip_rows=19)

    x = df[]