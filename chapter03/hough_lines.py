import cv2
import numpy as np

img = cv2.imread('../images/houghlines5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 120)

# cv2.HoughLinesP 函数的参数包括：
#
# image：输入图像，必须是二值图像，推荐使用 Canny 边缘检测的结果图像。
# rho：线段以像素为单位的距离精度，double 类型。
# theta：线段以弧度为单位的角度精度。
# threshold：累加平面的阈值参数，int 类型。
#   超过设定阈值才被检测出线段，值越大，基本上意味着检出的线段越长，检出的线段个数越少。
# minLineLength：线段以像素为单位的最小长度。
# maxLineGap：同一方向上两条线段判定为一条线段的最大允许间隔（断裂），
#   超过了设定值，则把两条线段当成一条线段，值越大，允许线段上的断裂越大，
#   越有可能检出潜在的直线段。
lines = cv2.HoughLinesP(edges, rho=1,
                        theta=np.pi/180.0,
                        threshold=20,
                        minLineLength=40,
                        maxLineGap=5)
for line in lines:
    x1, y1, x2, y2 = line[0]
    # 图像， pt1， pt2， 颜色[， 粗细[,线条的类型
    # cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("edges", edges)
cv2.imshow("lines", img)
cv2.waitKey()
cv2.destroyAllWindows()
