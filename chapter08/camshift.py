import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# 拍摄多帧以允许相机的自动曝光进行调整。
# 试着从摄像头捕捉10帧
for i in range(10):
    success, frame = cap.read()
if not success:
    exit(1)

# 在帧的中心定义一个初始的跟踪窗口
frame_h, frame_w = frame.shape[:2]
w = frame_w // 8
h = frame_h // 8
x = frame_w // 2 - w // 2
y = frame_h // 2 - h // 2
track_window = (x, y, w, h)

roi = frame[y:y + h, x:x + w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = None
# 计算初始窗口的归一化HSV直方图.
# 归一化HSV直方图是一种使用颜色信息来描述目标的方法，
# 它通过将每个像素的颜色值除以图像中像素的总数来归一化直方图。
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
# 归一化处理可以将图像的像素值范围缩放到一个特定的区间，
# cv2.NORM_MINMAX
# 使用线性归一化方法。线性归一化方法将数组中的最小值映射到alpha，将最大值映射到beta，其他值按比例缩放。
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# 定义了一个终止条件term_crit，它是一个元组，包含三个元素
# cv2.TERM_CRITERIA_COUNT表示迭代次数达到指定值时终止；
# cv2.TERM_CRITERIA_EPS表示结果在指定精度范围内时终止。
# 第二个元素是10，表示最大迭代次数为10。
# 第三个元素是1，表示精度阈值为1。
term_crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1)

success, frame = cap.read()
while success:

    # 将 HSV 直方图反向投影到帧上。
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # 使用CamShift执行跟踪。
    rotated_rect, track_window = cv2.CamShift(
        back_proj, track_window, term_crit)

    # 使用cv2.boxPoints函数寻找旋转跟踪矩形的顶点。
    box_points = cv2.boxPoints(rotated_rect)
    box_points = np.intp(box_points)
    # 使用cv2.polylines函数绘制连接这些顶点的线.
    cv2.polylines(frame, [box_points], True, (255, 0, 0), 2)

    # 展示反投影图像，它表示每个像素属于跟踪目标的概率。
    # 亮度越高，概率越大；亮度越低，概率越小
    cv2.imshow('back-projection', back_proj)
    cv2.imshow('camshift', frame)

    k = cv2.waitKey(1)
    if k == 27:  # Escape
        break

    success, frame = cap.read()
