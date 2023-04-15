import cv2
import numpy as np

OPENCV_MAJOR_VERSION = int(cv2.__version__.split('.')[0])

img = cv2.pyrDown(cv2.imread("../images/hammer.jpg"))

ret, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                            127, 255, cv2.THRESH_BINARY)

if OPENCV_MAJOR_VERSION >= 4:
    # OpenCV 4 or a later version is being used.
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
else:
    # OpenCV 3 or an earlier version is being used.
    # cv2.findContours has an extra return value.
    # The extra return value is the thresholded image, which (in
    # OpenCV 3.1 or an earlier version) may have been modified, but
    # we can ignore it.
    _, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
# contours轮廓
for c in contours:
    # 查找边界框坐标
    # 计算点集的直立边界矩形或灰度图像的非零像素。
    # 该函数计算并返回指定点集或灰度图像的非零像素的最小直立边框。
    x, y, w, h = cv2.boundingRect(c)
    # 图像， pt1， pt2， 颜色[， 粗细[
    # 画竖直水平的边框
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 查找最小面积
    rect = cv2.minAreaRect(c)
    # 计算最小面积矩形的坐标
    box = cv2.boxPoints(rect)
    # 将坐标规范化为整数
    box = np.intp(box)
    # 画轮廓
    # 图像， 轮廓，contourIdx 颜色[， 粗细[
    # contourIdx 指示要绘制的轮廓的参数。如果为负数，则绘制所有轮廓。
    # 画最小边框
    cv2.drawContours(img, [box], 0, (0, 0, 255), 3)

    # 计算最小包围圆的中心和半径
    (x, y), radius = cv2.minEnclosingCircle(c)
    # 强制转换为整数
    center = (int(x), int(y))
    radius = int(radius)
    # 画圆
    img = cv2.circle(img, center, radius, (255, 0, 0), 2)

# 轮廓
cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
cv2.imshow("contours", img)

cv2.waitKey()
cv2.destroyAllWindows()
