### This code reads multiple .ict files containing CFDC_N_INP data, extracts the latitude and longitude, and plots the data on a map using Cartopy and Matplotlib.
### 时间: 2025-08-05
### 作者: 付弘宇
import pathlib
import icartt
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# 文件夹路径
data_dir = pathlib.Path(r'E:\data\SOC\SOC-CFDC')
# 获取所有.ict文件
ict_files = list(data_dir.glob('*.ict'))

map_proj = ccrs.PlateCarree()
fig = plt.figure(dpi=300)
ax = fig.add_subplot(111, projection=map_proj)

extents = [100, 180, -80, -10]
ax.set_extent(extents, crs=map_proj)

ax.stock_img()
ax.coastlines()

# 聚合所有文件的数据
all_lon = []
all_lat = []
all_cfdc = []

for ict_file in ict_files:
    ict = icartt.Dataset(ict_file)
    cfdc = ict.data["CFDC_N_INP"]
    # 判断是否为标量（0-d array），如果是则转为列表
    if np.isscalar(cfdc) or np.ndim(cfdc) == 0:
        all_cfdc.append(float(cfdc))
    else:
        all_cfdc.extend(cfdc)

    # 提取经纬度
    lon = ict.data["LON"]
    lat = ict.data["LAT"]
    if np.isscalar(lon) or np.ndim(lon) == 0:
        all_lon.append(float(lon))
    else:
        all_lon.extend(lon)
    if np.isscalar(lat) or np.ndim(lat) == 0:
        all_lat.append(float(lat))
    else:
        all_lat.extend(lat)

vmin = min(all_cfdc)
vmax = max(all_cfdc)


# 绘制聚合后的数据
sc = ax.scatter(
        x=all_lon,
        y=all_lat,
        c=all_cfdc,
        cmap="Oranges",
        s=10,
        alpha=0.7,
        transform=map_proj,
        vmin=vmin,
        vmax=vmax
)

cbar = plt.colorbar(
        sc,
        # The ticks and format arguments are commented out because they are not currently needed.
        # ticks=[0, 100, 200, 300, 400, 500],
        # format="%d",
        label="INP Concentration (L⁻¹)", 
        orientation='vertical',
        fraction=0.046,
        pad=0.04,
        aspect=20,
        shrink=1.0,
        extend='neither',
        #ticks=[0, 100, 200, 300, 400, 500],
        #format="%d",
        location='right'
)

tick_proj = ccrs.PlateCarree()
ax.set_xticks(np.arange(100, 180 + 20, 20), crs=tick_proj)
ax.set_xticks(np.arange(100, 180 + 10, 10), minor=True, crs=tick_proj)
ax.set_yticks(np.arange(-70, -10 + 20, 20), crs=tick_proj)
ax.set_yticks(np.arange(-80, -10 + 10, 10), minor=True, crs=tick_proj)

ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

plt.show()
#plt.savefig("output_plot.png", dpi=300, bbox_inches="tight")