from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

# 定义已知数据点
x = np.array([0, 1, 2, 3])
y = np.array([0, 1, 4, 9])

# 创建插值函数
f = interp1d(x, y, kind='cubic')

# 计算插值结果
xnew = np.linspace(0, 3, num=100)
ynew = f(xnew)

# 绘制插值结果
plt.plot(x, y, 'o', xnew, ynew, '-')
plt.show()
