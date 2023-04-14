import numpy as np
import cv2 as cv


# 鼠标回调函数
def draw_circle(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img, (x, y), 100, (255, 0, 0), -1)


# 创建一个黑色图像，一个窗口，然后和回调绑定
img = np.zeros((512, 512, 3), np.uint8)
cv.namedWindow('image')
# 双击画圆
cv.setMouseCallback('image', draw_circle)
while 1:
    cv.imshow('image', img)
    if cv.waitKey(20) & 0xFF == 27:
        break
cv.destroyAllWindows()
