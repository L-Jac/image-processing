import cv2
import numpy as np

planets = cv2.imread("../images/planet_glow.jpg")
gray_img = cv2.cvtColor(planets, cv2.COLOR_BGR2GRAY)
gray_img2 = gray_img
gray_img3 = gray_img2
# 中值滤波器
gray_img = cv2.medianBlur(gray_img, 5)
cv2.imshow("HoughCircle1", gray_img)
g_hpf1 = gray_img2 - gray_img
cv2.imshow("HoughCircle3", g_hpf1)
blurred = cv2.GaussianBlur(gray_img2, (17, 17), 0)
cv2.imshow("HoughCircle4", blurred)
g_hpf2 = gray_img2 - gray_img
cv2.imshow("HoughCircle5", g_hpf2)
# 用霍夫梯度方法进行圆的检测。
# 首先对图像进行 Canny 边缘检测，对边缘中的每一个非 0 点，通过 Sobel 算法计算局部梯度。
# cv2.HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]])
# image：输入图像，需要灰度图。
# method：检测方法，常用 CV_HOUGH_GRADIENT。
# dp：检测内侧圆心的累加器图像的分辨率于输入图像之比的倒数。
# minDist：圆心之间的最小距离。
# param1：用于 Canny 边缘检测的较大阈值。
# param2：累加器阈值，越小，检测到的圆越多。
# minRadius：最小半径。
# maxRadius：最大半径。
circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT,
                           1, 120, param1=90, param2=40,
                           minRadius=0, maxRadius=0)

if circles is not None:
    # uint16	无符号整数（0 to 65535）
    # 将圆的坐标和半径四舍五入为最接近的整数
    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        # 画外圆
        # 图，圆心，半径，颜色，粗细
        cv2.circle(planets, (i[0], i[1]), i[2],
                   (0, 255, 0), 2)
        # 画圆心
        cv2.circle(planets, (i[0], i[1]), 2,
                   (0, 0, 255), 3)

cv2.imwrite("planets_hough_circles.jpg", planets)
cv2.imshow("HoughCircle2", planets)
cv2.waitKey()
cv2.destroyAllWindows()
