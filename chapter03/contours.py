import cv2
import numpy as np

# 获取 OpenCV 库的主版本号并将其存储在变量 OPENCV_MAJOR_VERSION 中。
OPENCV_MAJOR_VERSION = int(cv2.__version__.split('.')[0])

# Create a black image.
img = np.zeros((300, 300), dtype=np.uint8)

# Draw a square in two shades of gray.
img[50:150, 50:150] = 160
img[70:150, 70:150] = 128

# threshold 函数对图像进行阈值处理。
# 这会将图像中所有像素值大于 127 的像素设置为白色（255）
# 第四个参数 type 被设置为 0，这表示使用二进制阈值。
# 这意味着当像素值高于阈值时，它们将被设置为 maxval（在这个例子中为 255），否则将被设置为 0。
# ret 是实际使用的阈值（如果使用了自适应阈值，则与输入的阈值可能不同），
# thresh 是输出的二值图像。
ret, thresh = cv2.threshold(img, 127, 255, 0)

# 在阈值处理后的正方形上查找轮廓。
# TODO findContours函数
#   三个参数:输入图像，层次结构类型，轮廓近似方法
if OPENCV_MAJOR_VERSION >= 4:
    # 正在使用 OpenCV 4 或更高版本。
    # contours轮廓，hier层次结构
    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
else:
    # 正在使用 OpenCV 3 或更低版本。
    # cv2.findContours 有一个额外的返回值。
    # 额外的返回值是阈值处理后的图像，它（在 OpenCV 3.1 或更低版本中）
    # 可能已被修改，但我们可以忽略它。
    _, contours, hier = cv2.findContours(thresh, cv2.RETR_TREE,
                                         cv2.CHAIN_APPROX_SIMPLE)

# 彩色
color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# 先在彩色上绘图再读取坐标在img上绘图
img = cv2.drawContours(color, contours, -1, (0, 255, 0), 2)
cv2.imshow("contour1", color)
cv2.imshow("contour2", img)
cv2.waitKey()
cv2.destroyAllWindows()
