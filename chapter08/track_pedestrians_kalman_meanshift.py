import cv2
import numpy as np


class Pedestrian:
    """
    一个跟踪人体的类，含有ID，追踪框，直方图，卡尔曼滤波器
    """

    def __init__(self, id, hsv_frame, track_window):
        self.id = id
        # self.normal = normal
        self.track_window = track_window
        # 终止条件
        self.term_crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1)

        # 初始化直方图。
        x, y, w, h = track_window
        roi = hsv_frame[y:y + h, x:x + w]
        roi_hist = cv2.calcHist([roi], [0, 2], None, [15, 16],
                                [0, 180, 0, 256])
        self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255,
                                      cv2.NORM_MINMAX)

        # 初始化卡尔曼滤波器
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32) * 0.03
        cx = x + w / 2
        cy = y + h / 2
        self.kalman.statePre = np.array(
            [[cx], [cy], [0], [0]], np.float32)
        self.kalman.statePost = np.array(
            [[cx], [cy], [0], [0]], np.float32)

    def update(self, frame, hsv_frame):
        back_proj = cv2.calcBackProject(
            [hsv_frame], [0, 2], self.roi_hist, [0, 180, 0, 256], 1)

        ret, self.track_window = cv2.meanShift(
            back_proj, self.track_window, self.term_crit)
        x, y, w, h = self.track_window
        center = np.array([x + w / 2, y + h / 2], np.float32)

        # 使用卡尔曼滤波器的 predict 方法进行预测
        # prediction下一个状态
        prediction = self.kalman.predict()
        # 矫正
        estimate = self.kalman.correct(center)
        # 获取中心点的偏移量
        # python切片：[:2]表示获取前两个元素，
        # [:, 0] 表示获取二维数组中第一列的所有元素
        # 如果[:,:, 0] 表示获取三维数组中第一列的所有元素
        center_offset = estimate[:, 0][:2] - center
        # 根据中心点的偏移量来更新跟踪窗口的位置
        self.track_window = (x + int(center_offset[0]),
                             y + int(center_offset[1]), w, h)
        x, y, w, h = self.track_window

        # 中心画圆点
        cv2.circle(frame, (int(prediction[0]), int(prediction[1])),
                   4, (255, 0, 0), -1)

        # 画跟踪方框.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Draw the ID above the rectangle in blue text.
        cv2.putText(frame, f'ID: {self.id}', (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0),
                    1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture('../videos/pedestrians.avi')

    # 创造KNN背景差分器
    bg_subtractor = cv2.createBackgroundSubtractorKNN()
    # 历史长度设置20
    history_length = 20
    # 使用20帧来更新背景图像
    bg_subtractor.setHistory(history_length)

    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7))

    num_history_frames_populated = 0

    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break

        # 返回前景掩码
        fg_mask = bg_subtractor.apply(frame)

        # 检查是否已经处理了足够多的帧来建立背景模型。
        # 如果没有就增加 num_history_frames_populated 的值，并跳过当前帧。
        if num_history_frames_populated < history_length:
            num_history_frames_populated += 1
            continue

        # 阈值处理
        _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        # 形态学图像处理，先腐蚀再膨胀
        cv2.erode(thresh, erode_kernel, thresh, iterations=2)
        cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)

        # 找轮廓
        contours, hier = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 将图像从 BGR 颜色空间转换为 HSV 颜色空间。
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        pedestrians = []
        # should_initialize_pedestrians 表示是否应该初始化行人列表。
        # 如果它的值为 True，那么就表示行人列表为空，需要初始化；
        # 否则，就表示行人列表已经初始化过了。
        should_initialize_pedestrians = len(pedestrians) == 0
        id = 1
        for c in contours:

            if cv2.contourArea(c) > 500:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (0, 0, 255), 5)
                if should_initialize_pedestrians:
                    pedestrians.append(
                        Pedestrian(id, hsv_frame, (x, y, w, h)))
                    id += 1

        # Update the tracking of each pedestrian.
        for pedestrian in pedestrians:
            pedestrian.update(frame, hsv_frame)

        cv2.imshow('Pedestrians Tracked', frame)

        k = cv2.waitKey(110)
        if k == 27:  # Escape
            break


if __name__ == "__main__":
    main()
