import cv2
import numpy
import os

# 创建了一个长度为 120000 的随机字节数组.
randomByteArray = bytearray(os.urandom(120000))
flatNumpyArray = numpy.array(randomByteArray)

# 重塑为一个 300x400 的灰度图像并保存为 “RandomGray.png”.
grayImage = flatNumpyArray.reshape(300, 400)
cv2.imwrite('RandomGray.png', grayImage)

# 重塑为一个 100x400x3 的彩色图像并保存为 “RandomColor.png”.
bgrImage = flatNumpyArray.reshape(100, 400, 3)
cv2.imwrite('RandomColor.png', bgrImage)
