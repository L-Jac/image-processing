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

black = np.zeros_like(img)
for cnt in contours:
    # 计算等值线周长或曲线长度。
    # 该函数计算曲线长度或闭合等高线周长。
    # 参数true表示曲线闭合
    # epsilon 是用来控制近似轮廓的精度的。它的值越小，近似轮廓就越接近原始轮廓。
    # 如果把0.01换成0.001将会更接近原始轮廓
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    # 使用道格拉斯-普克算法（Douglas-Peucker algorithm）来近似轮廓。
    # 该函数接受一个轮廓和一个精度参数 epsilon，并返回一个近似轮廓。
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    # 用来计算一个点集或轮廓的凸包
    # 该函数接受一个点集或轮廓，并返回一个表示凸包的点集。
    hull = cv2.convexHull(cnt)
    cv2.drawContours(black, [cnt], -1, (0, 255, 0), 2)
    cv2.drawContours(black, [approx], -1, (255, 255, 0), 2)
    cv2.drawContours(black, [hull], -1, (0, 0, 255), 2)

cv2.imshow("hull", black)
cv2.waitKey()
cv2.destroyAllWindows()
