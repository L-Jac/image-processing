import cv2
import numpy as np
from matplotlib import pyplot as plot

# Numpy中的傅立叶变换
# cv2.IMREAD_GRAYSCALE：以灰度模式加载图片，可以直接写0，彩色写1
img = cv2.imread('../images/bb.jpg', 0)
# 计算二维的傅里叶变换
# 第一个参数是输入图像，即灰度图像。第二个参数是可选的，它决定输出数组的大小
f = np.fft.fft2(img)
# 将零频率分量移动到频谱的中心
fshift = np.fft.fftshift(f)
# 计算了傅里叶变换后图像的幅度谱。
# 它首先取绝对值，然后取对数，最后乘以20。
# 这样做的目的是为了更好地显示结果，因为傅里叶变换的结果通常具有很大的动态范围。
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# 这部分代码的目的是创建一个掩码，用来屏蔽掉频谱中心的一部分。
# 首先，它计算出图像的中心点坐标（crow，ccol）。
# 然后，它在频谱中心周围创建一个大小为60x60的矩形区域，并将该区域内的值都设为0。
# 这样做可以去除图像中的低频分量，从而实现高通滤波的效果。
row, cols = img.shape
crow, ccol = row // 2, cols // 2
fshift[crow - 30: crow + 30, ccol - 30: ccol + 30] = 0

# 逐步傅里叶逆变换还原图像得到HPF
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

img2 = cv2.medianBlur(img, 5)
img3 = img - img2
cv2.imshow("window1", img)
cv2.imshow("window2", img3)
cv2.imshow("window3", img_back)
plot.subplot(221), plot.imshow(img, cmap="gray")
plot.title("Input"), plot.xticks([]), plot.yticks([])

plot.subplot(222), plot.imshow(magnitude_spectrum, cmap="gray")
plot.title('magnitude_spectrum'), plot.xticks([]), plot.yticks([])

plot.subplot(223), plot.imshow(img_back, cmap="gray")
plot.title("Input in JET"), plot.xticks([]), plot.yticks([])
plot.show()
