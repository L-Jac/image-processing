import numpy as np
import cv2

# minDisparity：最小视差值。默认值为 0。
minDisparity = 16
# numDisparities：视差范围。必须是 16 的倍数。
numDisparities = 192 - minDisparity
# blockSize：匹配块大小。必须是奇数。
blockSize = 5
# uniquenessRatio：视差唯一性百分比。默认值为 0。
uniquenessRatio = 1
# speckleWindowSize：散斑窗口大小。默认值为 0。
speckleWindowSize = 3
# speckleRange：散斑范围。默认值为 0。
speckleRange = 3
# disp12MaxDiff：左右视差图的最大差异。默认值为 0。
disp12MaxDiff = 200
# P1，P2：控制视差平滑度的惩罚系数。
P1 = 600
P2 = 2400

stereo = cv2.StereoSGBM_create(
    minDisparity=minDisparity,
    numDisparities=numDisparities,
    blockSize=blockSize,
    uniquenessRatio=uniquenessRatio,
    speckleRange=speckleRange,
    speckleWindowSize=speckleWindowSize,
    disp12MaxDiff=disp12MaxDiff,
    P1=P1,
    P2=P2
)

imgL = cv2.imread('../images/color1_small.jpg')
imgR = cv2.imread('../images/color2_small.jpg')


def update(sliderValue=0):
    try:
        # cv2.getTrackbarPos：轨道栏名称，窗口名
        # 返回对应栏的值，用于更新参数
        blockSize = cv2.getTrackbarPos('blockSize', 'Disparity')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'Disparity')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'Disparity')
        speckleRange = cv2.getTrackbarPos('speckleRange', 'Disparity')
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'Disparity')
    except cv2.error:
        # 一个或多个滑块尚未创建。
        return

    # 这段代码使用之前从轨道栏获取的值更新立体匹配对象 stereo 的参数
    stereo.setBlockSize(blockSize)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setSpeckleRange(speckleRange)
    stereo.setDisp12MaxDiff(disp12MaxDiff)

    # 使用 stereo.compute 函数计算两个图像之间的视差。
    # 这个函数返回一个表示视差的矩阵。由于视差矩阵中的值是以 16 为单位的，所以需要将其除以 16.0 来获得真实的视差值。
    # astype(np.float32) 是将矩阵中的元素类型转换为浮点数，以便进行除法运算。
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    cv2.imshow('Left', imgL)
    cv2.imshow('Right', imgR)
    # 对视差矩阵进行归一化处理，以便在显示时更好地观察视差图。这样处理后，视差矩阵中的值将在 0 到 1 之间。
    cv2.imshow('Disparity', (disparity - minDisparity) / numDisparities)


# 创建窗口
cv2.namedWindow('Disparity')
# 生成状态栏，触发函数设为update
cv2.createTrackbar('blockSize', 'Disparity', blockSize, 21, update)
cv2.createTrackbar('uniquenessRatio', 'Disparity', uniquenessRatio, 50, update)
cv2.createTrackbar('speckleWindowSize', 'Disparity', speckleWindowSize, 200, update)
cv2.createTrackbar('speckleRange', 'Disparity', speckleRange, 50, update)
cv2.createTrackbar('disp12MaxDiff', 'Disparity', disp12MaxDiff, 250, update)

# Initialize the disparity map. Show the disparity map and images.
update()

# Wait for the user to press any key.
# Meanwhile, update() will be called anytime the user moves a slider.
cv2.waitKey()
