import cv2

# 定义blur、erode和dilate运算的核的大小
BLUR_RADIUS = 21
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

cap = cv2.VideoCapture(0)
# 拍摄多帧以允许相机的自动曝光进行调整。
# 试着从摄像头捕捉10帧
for i in range(10):
    success, frame = cap.read()
if not success:
    exit(1)

# 第10帧图像转换为灰度图像
gray_background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 对其进行高斯模糊
gray_background = cv2.GaussianBlur(gray_background,
                                   (BLUR_RADIUS, BLUR_RADIUS), 0)

success, frame = cap.read()
while success:
    # 对每一帧的都进行灰度转换和高斯模糊
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame,
                                  (BLUR_RADIUS, BLUR_RADIUS), 0)

    # 分离出运动目标，也就是前景
    # 比较背景帧和当前帧
    diff = cv2.absdiff(gray_background, gray_frame)
    # 阈值化处理
    _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
    # cv2.erode 和 cv2.dilate 函数对二值图像进行形态学处理，以去除噪声并填补空洞。
    cv2.erode(thresh, erode_kernel, thresh, iterations=2)
    cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)

    # 使用cv2.findContours 函数来在二值图像 thresh 中查找轮廓。
    # cv2.RETR_EXTERNAL 参数表示只检测最外层的轮廓
    # cv2.CHAIN_APPROX_SIMPLE 参数表示使用简单的轮廓近似方法来压缩水平、垂直和对角方向的轮廓。
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    # contours 是一个列表，其中包含了图像中所有轮廓的坐标点。
    # hier 是一个数组，其中包含了轮廓之间的层次关系信息。


    """
    OPENCV_MAJOR_VERSION = int(cv2.__version__.split('.')[0])
    if OPENCV_MAJOR_VERSION >= 4:
        # OpenCV 4 or a later version is being used.
        contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
    else:
        # OpenCV 3 or an earlier version is being used.
        # cv2.findContours has an extra return value.
        # The extra return value is the thresholded image, which is
        # unchanged, so we can ignore it.
        _, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
    """

    for c in contours:
        # cv2.contourArea 函数计算每个轮廓的面积
        if cv2.contourArea(c) > 4000:
            # cv2.boundingRect 函数计算该轮廓的外接矩形
            x, y, w, h = cv2.boundingRect(c)
            # 画矩形
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # diff 窗口显示的是当前帧与背景帧之间的差异图像
    cv2.imshow('diff', diff)
    # thresh 窗口显示的是经过阈值化处理后的二值图像
    # 白色的部分表示前景，黑色的部分表示背景。
    cv2.imshow('thresh', thresh)
    cv2.imshow('detection', frame)

    k = cv2.waitKey(1)
    if k == 27:  # Escape
        break

    success, frame = cap.read()
