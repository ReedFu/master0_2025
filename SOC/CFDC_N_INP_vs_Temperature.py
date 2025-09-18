### This script reads multiple .ict files, extracts temperature and CFDC_N_INP data, and plots them.
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
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("CFDC_N_INP (L^-1)")
ax.set_title("CFDC_N_INP vs Temperature")
ax.set_xscale("linear")
ax.grid(True)

# 聚合所有文件的数据
all_Temp = []
all_cfdc = []

for ict_file in ict_files:
    ict = icartt.Dataset(ict_file)
    cfdc = ict.data["CFDC_N_INP"]
    # 判断是否为标量（0-d array），如果是则转为列表
    if np.isscalar(cfdc) or np.ndim(cfdc) == 0:
        all_cfdc.append(float(cfdc))
    else:
        all_cfdc.extend(cfdc)
    # 判断是否为标量（0-d array），如果是则转为列表
    Temp = ict.data["CFDC_Temp"]
    if np.isscalar(Temp) or np.ndim(Temp) == 0:
        all_Temp.append(float(Temp))
    else:
        all_Temp.extend(Temp)

# 绘制聚合后的数据
sc = ax.scatter(
        x=all_Temp,
        y=all_cfdc,
)


plt.show()
#plt.savefig("output_plot.png", dpi=300, bbox_inches="tight")