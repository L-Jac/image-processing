import cv2

# 常规读取，灰度图转换
img = cv2.imread('../images/varese.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 创建SIFT检测对象,一个类的实例
sift = cv2.SIFT_create()
# 计算灰度图像的特征和描述符：
# 该对象使用DoG检测关键点，再计算每个关键点周围区域的特征向量
# 返回值是一个元组，包含一个关键点列表和另一个关键点的描述符列表。
keypoints, descriptors = sift.detectAndCompute(gray, None)
# keypoints:
#   关键点列表中的每个元素都是一个cv2.KeyPoint类的实例，它们表示了图像中检测到的关键点。
#   具有以下属性：
#   ·pt（点）属性包括图像中关键点的x和y坐标。
#   ·size属性表示特征的直径。
#   ·angle属性表示特征的方向，如前面处理过的图像中的径向线所示。
#   ·response属性表示关键点的强度。
#       由SIFT分类的一些特征比其他特征更强，response可以评估特征强度。
#   ·octave属性表示发现该特征的图像金字塔层。
#       SIFT算法的操作方式类似于人脸检测算法，迭代处理相同的图像，但是每次迭代时都会更改输入。
#       具体来说，图像尺度是在算法每次迭代（octave）时都变化的一个参数。
#       因此，octave属性与检测到关键点的图像尺度有关。
#   ·class_id属性可以用来为一个关键点或者一组关键点分配自定义的标识符。

cv2.drawKeypoints(img, keypoints, img, (51, 163, 236),
                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('sift_keypoints', img)
cv2.waitKey()
