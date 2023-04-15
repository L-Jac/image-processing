import cv2
import numpy as np
from scipy import ndimage

# 定义两个卷积核
kernel_3x3 = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])

kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                       [-1,  1,  2,  1, -1],
                       [-1,  2,  4,  2, -1],
                       [-1,  1,  2,  1, -1],
                       [-1, -1, -1, -1, -1]])

img = cv2.imread("../images/statue_small.jpg",
                 cv2.IMREAD_GRAYSCALE)
# HPF方法一，使用锐化卷积核卷积增强图像的边缘和细节
# Scipy的ndimage模块提供convolve函数实现多维卷积
k3 = ndimage.convolve(img, kernel_3x3)
k5 = ndimage.convolve(img, kernel_5x5)

# HPF法二，高斯模糊处理后(得到LPF，使用均值滤波器亦可)再减去，消除低频分量
# 使用cv2.GaussianBlur函数对图像进行高斯模糊处理
blurred = cv2.GaussianBlur(img, (17, 17), 0)
# 原始图像减去高斯模糊得到高通滤波器
# cv2.GaussianBlur函数的原型为：
# cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]])。
# 其中src为输入图像，ksize为高斯内核大小，
# sigmaX为X方向上的高斯核标准偏差，sigmaY为Y方向上的高斯核标准偏差。
# 如果sigmaY为零，则将其设置为等于sigmaX；
# 如果两个西格玛均为零，则分别根据ksize.width和ksize.height进行计算2。
g_hpf = img - blurred

cv2.imshow("3x3", k3)
cv2.imshow("5x5", k5)
cv2.imshow("blurred", blurred)
cv2.imshow("g_hpf", g_hpf)
# cv2.waitKey()用于暂停程序，直到用户按下任意键。
# 这样，用户就可以查看处理后的图像，然后再按任意键关闭图像窗口并继续执行程序。
cv2.waitKey()
cv2.destroyAllWindows()
# 第一种方法可以快速地增强图像中的边缘和细节，但它也会增强图像中的噪声。
# 如果你希望快速地增强图像中的边缘和细节，而不太关心噪声的问题，那么第一种方法可能更适合你。
# 第二种方法可以更好地消除图像中的噪声，但它也会消除一些细节。
# 如果你希望在保留边缘和细节的同时消除噪声，那么第二种方法可能更适合你。

# TODO 高斯模糊与中值模糊的异同
#   高斯模糊是一种线性滤波器，它通过对图像进行加权平均来实现模糊效果。
#   权重是根据高斯分布函数计算的，因此距离中心点越近的像素权重越大，越远离中心的像素权重越小1。
#   中值模糊是一种非线性滤波器，它通过对图像中每个像素周围的像素进行排序，
#   然后取中间值作为该像素的新值来实现模糊效果。它对椒盐噪声有很好的抑制作用2。
