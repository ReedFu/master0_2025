import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_table(r"D:\Data\PAMARCMiP\Hartmann-etal_2019_INP-PAMARCMIP\datasets\P5_210_PAMARCMIP_2018_1803230301_INP.tab", skiprows=19)

fig = plt.figure(dpi=500)
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Temperature(℃)')
ax.set_ylabel('INP concentration(#/L)')
ax.tick_params(which='both', top=True, bottom=True, left=True, right=True, direction='in')
ax.set_title('INP concentration vs Temperature')

x = df[df.columns[2]]
y = df[df.columns[3]]
ax.scatter(x,y)

fig.savefig("./master0_2025/PAMARCMiP/2.png", bbox_inches='tight')