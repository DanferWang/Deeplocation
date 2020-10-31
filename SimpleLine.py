import matplotlib.pyplot as mpl
import csv
import pandas as pd
import math

values = []
distance = []
gcd = pd.read_csv('demo/data/face_less0_ten_result_3000.csv', usecols = ["great_circle_distance"])
gcd =gcd.astype(float)
for i in range(len(gcd)):
    values.append(i)
    distance.append(gcd.iloc[i])

## 画线：polt()
mpl.plot(values, distance, linewidth=3)  ## 调粗细
mpl.title("Square Function")  ## 加标签
# 设置坐标轴信息
mpl.xlabel("X", fontsize=15)
mpl.ylabel("Y=X**2", fontsize=15)
mpl.axis([0,3000, 0.1,12742])
mpl.yscale('log', base=math.e)
mpl.tick_params(axis="both", labelsize=15)  ## 设置坐标轴粗细

## 保存图像
# mpl.savefig("Square_Function.png",bbox_inches="tight")

## 显示图像
mpl.show()