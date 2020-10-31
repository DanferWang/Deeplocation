import matplotlib.pyplot as mpl

values = []
squares = []

for i in range(1000):
    values.append(i)
    squares.append(i ** 2)

## 实参s调整点的大小,edgecolors调整点边缘颜色，c调整点的填充颜色（也可用RGB）
mpl.scatter(values, squares, c="red", s=100, edgecolors="yellow",marker="*")
mpl.title("Square Number")  ## 加标签
## 设置坐标轴信息
mpl.xlabel("X", fontsize=15)
mpl.ylabel("Y=X**2", fontsize=15)
## 设置坐标轴取值范围
mpl.axis([0, 1050, 0, 1000000])
mpl.tick_params(axis="both", labelsize=15)  ## 设置坐标轴粗细

## 颜色映射
squares.reverse()
mpl.scatter(values, squares, c=squares, cmap=mpl.cm.Blues)
mpl.show()
