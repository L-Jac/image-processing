import cv2
import numpy as np

# OpenCV的Canny边缘检测算法
# 1，使用高斯滤波平滑图像，减少图像中噪声。
# 2，计算图像中每个像素的梯度方向和幅值。
# 3，应用非极大值抑制算法消除边缘检测带来的杂散响应。
# 4，应用双阈值法划分强边缘和弱边缘。
# 5，消除孤立的弱边缘。
img = cv2.imread("../images/statue_small.jpg", cv2.IMREAD_GRAYSCALE)
canny_img = cv2.Canny(img, 200, 300)
cv2.imshow("canny", canny_img)
cv2.waitKey()
cv2.destroyAllWindows()
