# 处理文件，摄像头，GUI
import cv2
import numpy as np

# TODO 读取写入图像文件
# 生成3*3全0像素数组(像素值0~225)
img = np.zeros((3, 3), dtype=np.uint8)

# cv2
# ·cv2.IMREAD_COLOR：该模式是默认选项，提供3通道的BGR图像，每个通道一个8位值（0～255）。
# ·cv2.IMREAD_GRAYSCALE：该模式提供8位灰度图像。
# ·cv2.IMREAD_ANYCOLOR：该模式提供每个通道8位的BGR图像或者8位灰度图像，具体取决于文件中的元数据。
# ·cv2.IMREAD_UNCHANGED：该模式读取所有的图像数据，包括作为第4通道的α或透明度通道（如果有的话）。
# ·cv2.IMREAD_ANYDEPTH：该模式加载原始位深度的灰度图像。
#     例如，如果文件以这种格式表示一幅图像，那么它提供每个通道16位的一幅灰度图像。
# ·cv2.IMREAD_ANYDEPTH｜cv2.IMREAD_COLOR：该组合模式加载原始位深度的BGR彩色图像。
# ·cv2.IMREAD_REDUCED_GRAYSCALE_2：该模式加载的灰度图像的分辨率是原始分辨率的1/2。
#     例如，如果文件包括一幅640×480的图像，那么它加载的是一幅320×240的图像。
# ·cv2.IMREAD_REDUCED_COLOR_2：该模式加载每个通道8位的BGR彩色图像，分辨率是原始图像的1/2。
# ·cv2.IMREAD_REDUCED_GRAYSCALE_4：该模式加载灰度图像，分辨率是原始图像的1/4。
# ·cv2.IMREAD_REDUCED_COLOR_4：该模式加载每个通道8位的彩色图像，分辨率是原始图像的1/4。
# ·cv2.IMREAD_REDUCED_GRAYSCALE_8：该模式加载灰度图像，分辨率是原始图像的1/8。
# ·cv2.IMREAD_REDUCED_COLOR_8：该模式加载每个通道8位的彩色图像，分辨率为原始图像的1/8。

# cv2.cvtColor转成BGR格数
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
print(img)
# shape返回行，列，通道数
print(img.shape)
# cv2.imwrite转换格式
# image = cv2.imread('cute.jpg')
# cv2.imwrite('Cute.png', image)
