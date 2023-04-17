import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('../images/5_of_diamonds.png')
# 以灰度图的形式读取
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 对图像进行阈值处理。
# cv2.THRESH_BINARY_INV 表示使用反向二进制阈值处理。
# 在这种情况下，如果像素值大于阈值，则将其设置为 0（黑色），否则将其设置为最大值（通常为 255，白色）。
# cv2.THRESH_OTSU 表示使用 Otsu’s 方法自动计算阈值。Otsu’s 方法是一种用于自动确定阈值的方法，它通过计算图像直方图来确定最佳阈值。
# 所以不一定大于0就变白，因为阈值由 Otsu’s 方法得出
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# 消去噪声
kernel = np.ones((3, 3), np.uint8)
# 使用形态学开运算对阈值处理后的图像进行了两次迭代，以消除噪声。
# 形态学开运算是一种用于消除小的白色噪点的图像处理操作。
# 它首先对图像进行腐蚀操作去除边缘毛躁，然后再进行膨胀操作。腐蚀操作会消除小的白色噪点，而膨胀操作则会使剩余的白色区域扩大。
# 开运算、闭运算、形态学梯度、顶帽操作和底帽操作
# MORPH_OPEN
# MORPH_CLOSE
# MORPH_CLOSEorphological
# MORPH_TOPHAT
# MORPH_BLACKHAT
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# 使用形态学膨胀操作找到确定的背景区域，相当于加粗
# iterations – 膨胀次数，默认为1
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# 距离变换和阈值处理找到确定的前景区域
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg = sure_fg.astype(np.uint8)

# 使用 cv2.subtract() 函数从确定的背景区域中减去确定的前景区域，以找到未知区域
unknown = cv2.subtract(sure_bg, sure_fg)

# 使用 cv2.connectedComponents() 函数对确定的前景区域进行连通分量标记。
# 这会为每个连通的前景对象分配一个唯一的标签。
ret, markers = cv2.connectedComponents(sure_fg)

# 向所有标签添加一个，以确保背景不是 0，而是 1。
markers += 1

# 将未知区域的标签设置为 0
markers[unknown == 255] = 0

# 使用 cv2.watershed() 函数对图像进行分水岭分割，并将分割结果存储在 markers 变量中。
markers = cv2.watershed(img, markers)
# 水岭算法会根据给定的标记图像来确定图像中每个像素属于哪个区域。在分割结果中，边界像素的标签为 -1。
# 因此，这段代码将边界像素的颜色设置为蓝色（[255, 0, 0]）
img[markers == -1] = [255, 0, 0]

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
