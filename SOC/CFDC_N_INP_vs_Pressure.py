### SOC飞机观测数据处理 ###
### 读取多个 .ict 文件, 绘制 INP 浓度与气压(表征高度)的散点图 ###
### 时间: 2025-08-06
### 作者: 付弘宇
import pathlib
import icartt
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# 文件夹路径
data_dir = pathlib.Path(r'E:\data\SOC\SOC-CFDC')
# 获取所有.ict文件
ict_files = list(data_dir.glob('*.ict'))

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
ax.set_xlabel("CFDC_Pressure (hPa)")
ax.set_ylabel("CFDC_N_INP (L^-1)")
ax.set_title("CFDC_N_INP vs Pressure")
ax.set_xscale("linear")
ax.grid(True)

# 聚合所有文件的数据
all_Pressure = []
all_cfdc = []

for ict_file in ict_files:
    ict = icartt.Dataset(ict_file)
    cfdc = ict.data["CFDC_N_INP"]
    # 判断是否为标量（0-d array），如果是则转为列表
    if np.isscalar(cfdc) or np.ndim(cfdc) == 0:
        all_cfdc.append(float(cfdc))
    else:
        all_cfdc.extend(cfdc)
    # 
    Pressure = ict.data["CFDC_Pressure"]
    if np.isscalar(Pressure) or np.ndim(Pressure) == 0:
        all_Pressure.append(float(Pressure))
    else:
        all_Pressure.extend(Pressure)

# 绘制聚合后的数据
sc = ax.scatter(
        x=all_Pressure,
        y=all_cfdc,
)


plt.show()
#plt.savefig("output_plot.png", dpi=300, bbox_inches="tight")