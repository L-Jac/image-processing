# 基本背景差分的修改
# （1）用MOG背景差分器替换基本背景差分模型。
# （2）使用视频文件（而不是摄像头）作为输入。
# （3）取消高斯模糊。
# （4）调整阈值、形态学和轮廓分析步骤中使用的参数。
import cv2

# cv2.BackgroundSubtractorMOG2的参数
# history：用于训练背景的帧数，默认为500帧。
# varThreshold：方差阈值，用于判断当前像素是前景还是背景。默认值为16。
# detectShadows：是否检测影子，设为True为检测，False为不检测。默认值为True。
# 初始化一个MOG2背景差分器
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# 定义erode和dilate运算的核的大小
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

cap = cv2.VideoCapture('../videos/hallway.mpg')
success, frame = cap.read()
while success:

    # 使用bg_subtractor.apply方法对每一帧应用MOG2背景减除器，得到前景掩码fg_mask
    fg_mask = bg_subtractor.apply(frame)

    # 阈值化处理
    _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
    # 使用cv2.erode和cv2.dilate函数对二值图像进行腐蚀和膨胀操作
    # 第三个参数是输出图像，第四个参数是迭代次数
    # 腐蚀操作可以去除小的白色噪点，而膨胀操作可以填补小的黑色空洞
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
        if cv2.contourArea(c) > 1000:
            # cv2.boundingRect 函数计算该轮廓的外接矩形
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

    cv2.imshow('mog', fg_mask)
    cv2.imshow('thresh', thresh)
    # 通过调用bg_subtractor.getBackgroundImage()方法
    # 返回背景减除器内部模型的背景图像
    cv2.imshow('background', bg_subtractor.getBackgroundImage())
    cv2.imshow('detection', frame)

    k = cv2.waitKey(30)
    if k == 27:  # Escape
        break

    success, frame = cap.read()
cap.release()
cv2.destroyAllWindows()
