import os

import cv2
import numpy


def read_images(path, image_size):
    names = []
    training_images, training_labels = [], []
    label = 0
    for dirname, subdirnames, filenames in os.walk(path):
        for subdirname in subdirnames:
            names.append(subdirname)
            # 使用 os.path.join 函数将当前子目录的路径和文件名拼接起来，以得到完整的文件路径。
            subject_path = os.path.join(dirname, subdirname)
            # os.listdir 函数来遍历指定目录下的所有文件。
            for filename in os.listdir(subject_path):
                img = cv2.imread(os.path.join(subject_path, filename),
                                 cv2.IMREAD_GRAYSCALE)
                # 如果读取失败，则跳过该文件。
                if img is None:
                    continue
                img = cv2.resize(img, image_size)
                training_images.append(img)
                training_labels.append(label)
            label += 1
    # 将 training_images 和 training_labels 列表转换为 NumPy 数组
    training_images = numpy.asarray(training_images, numpy.uint8)
    training_labels = numpy.asarray(training_labels, numpy.int32)
    return names, training_images, training_labels


path_to_training_images = '../data/at'
training_image_size = (200, 200)
names, training_images, training_labels = read_images(
    path_to_training_images, training_image_size)

# 用EigenFaceRecognizer_create()创建一个人脸识别器
# 其他模型FisherFaceRecognizer_create()，LBPHFaceRecognizer_create()
model = cv2.face.EigenFaceRecognizer_create()
model.train(training_images, training_labels)

face_cascade = cv2.CascadeClassifier(
    f'{cv2.data.haarcascades}haarcascade_frontalface_default.xml')

camera = cv2.VideoCapture(0)
while cv2.waitKey(1) == -1:
    success, frame = camera.read()
    if success:
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 提取人脸区域
            roi_gray = gray[x:x+w, y:y+h]
            if roi_gray.size == 0:
                # 脸部区域像素为空。也许脸在图像边缘。
                # 跳过它。
                continue
            roi_gray = cv2.resize(roi_gray, training_image_size)
            label, confidence = model.predict(roi_gray)
            text = f'{names[label]}, confidence={confidence:.2f}'
            cv2.putText(frame, text, (x, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Recognition', frame)
