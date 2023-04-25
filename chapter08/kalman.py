import cv2
import numpy as np

# 初始化一幅黑色图像
img = np.zeros((800, 800, 3), np.uint8)

# 初始化一个卡尔曼滤波器
# 第1个参数是卡尔曼滤波器跟踪（或预测）的变量数量，
# x位置、y位置、x速度以及y速度。
# 第2个参数是提供给卡尔曼滤波器作为测量值的变量数量，
# x位置和y位置。
kalman = cv2.KalmanFilter(4, 2)
# 测量矩阵被定义为2x4的矩阵。用于将状态向量映射到测量向量
kalman.measurementMatrix = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0]], np.float32)
# 转移矩阵被定义为4x4的矩阵。用于预测下一个状态向量
kalman.transitionMatrix = np.array(
    [[1, 0, 1, 0],
     [0, 1, 0, 1],
     [0, 0, 1, 0],
     [0, 0, 0, 1]], np.float32)
# 过程噪声协方差矩阵被定义为一个4x4的矩阵，并乘以0.03。
# 过程噪声协方差矩阵描述了状态转移过程中的不确定性。
# 它用于更新卡尔曼滤波器的误差协方差矩阵。
kalman.processNoiseCov = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]], np.float32) * 0.03

last_measurement = None
last_prediction = None


def on_mouse_moved(event, x, y, flags, param):
    global img, kalman, last_measurement, last_prediction

    measurement = np.array([[x], [y]], np.float32)
    # 检查全局变量 last_measurement 是否为 None。
    # 如果是，则表示这是第一次测量
    if last_measurement is None:
        # 卡尔曼滤波器的状态被更新以匹配测量值。预测值被设置为测量值。
        # 先验状态和后验状态
        # x位置、y位置、x速度以及y速度。(鼠标初位置，速度0)
        kalman.statePre = np.array(
            [[x], [y], [0], [0]], np.float32)
        kalman.statePost = np.array(
            [[x], [y], [0], [0]], np.float32)
        prediction = measurement
    else:
        # 使用测量值对卡尔曼滤波器进行校正
        kalman.correct(measurement)
        # 使用卡尔曼滤波器的 predict 方法进行预测
        prediction = kalman.predict()  # Gets a reference, not a copy

        # 在图像上绘制从上一个测量值到当前测量值的绿色线条，
        cv2.line(img, (int(last_measurement[0]), int(last_measurement[1])),
                 (int(measurement[0]), int(measurement[1])), (0, 255, 0))

        # 并绘制从上一个预测值到当前预测值的红色线条。
        cv2.line(img, (int(last_prediction[0]), int(last_prediction[1])),
                 (int(prediction[0]), int(prediction[1])), (0, 0, 255))

    # 更新全局变量 last_prediction 和 last_measurement 的值
    last_prediction = prediction.copy()
    last_measurement = measurement


cv2.namedWindow('kalman_tracker')
# 调用鼠标回调函数on_mouse_moved
cv2.setMouseCallback('kalman_tracker', on_mouse_moved)

while True:
    cv2.imshow('kalman_tracker', img)
    k = cv2.waitKey(1)
    if k == 27:  # Escape
        cv2.imwrite('kalman.png', img)
        break
