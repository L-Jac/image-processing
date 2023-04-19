import cv2


# 加载haar级联分类器，frontalface正面人脸
face_cascade = cv2.CascadeClassifier(
    f'{cv2.data.haarcascades}haarcascade_frontalface_default.xml')
# 以灰度图的形式读取
img = cv2.imread('../images/woodcutters.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 使用级联分类器的detectMultiScale方法在灰度图像中检测人脸。
# 该方法返回一个包含人脸矩形坐标的元组列表
# scaleFactor被设置为1.08，这意味着图像尺寸将在每个尺度上缩小8％；
# minNeighbors被设置为5，这意味着只有当一个候选矩形周围有至少5个其他候选矩形时，它才会被保留
# minSize和maxSize：指定可检测对象的最小和最大尺寸。这些参数用来限制检测到的对象的尺寸范围。
faces = face_cascade.detectMultiScale(gray, 1.08, 5)
# 遍历矩形并写入原始图像
for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
  
cv2.imshow('Woodcutters Detected!', img)
cv2.imwrite('./woodcutters_detected.png', img)
cv2.waitKey(0)
