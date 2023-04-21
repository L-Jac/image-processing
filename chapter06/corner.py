import cv2

# TODO cornerHarris用于检测图像中的角点
# 常规导入，转成灰度图
img = cv2.imread('../images/chess_board.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.cornerHarris
# src：输入图像，数据类型为float32，且需要为单通道8位或者浮点型图像。
# blockSize：表示邻域的大小。
# ksize：表示Sobel算子的孔径大小。3~31间的奇数表示对角点的灵敏度
# k：Harris参数。
# borderType：图像像素的边界模式。注意它有默认值BORDER_DEFAULT。
dst = cv2.cornerHarris(gray, 2, 27, 0.04)
# cv2.cornerHarris返回浮点格式的图像。返回灰度图，灰度值表示分值
# 该图像中的每个值表示源图像对应像素的一个分值。
# 中等的分值或者高的分值表明像素很可能是一个角点。分值最低的像素可以视为非角点。
# 考虑下面的代码行：
# 我们选取的像素的分值至少是最高分值的1%
img[dst > 0.01 * dst.max()] = [0, 0, 255]
cv2.imshow('corners', img)
cv2.waitKey()
