import cv2
import os


output_folder = '../data/at/test2'
# 如果不存在就创建这个文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

face_cascade = cv2.CascadeClassifier(
    f'{cv2.data.haarcascades}haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    f'{cv2.data.haarcascades}haarcascade_eye.xml')

camera = cv2.VideoCapture(0)
count = 0
while cv2.waitKey(1) == -1:
    success, frame = camera.read()
    if success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 1.3, 5, minSize=(120, 120))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # 重塑成200*200
            face_img = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            face_filename = f'{output_folder}/{count}.pgm'
            cv2.imwrite(face_filename, face_img)
            count += 1
        cv2.imshow('Capturing Faces...', frame)
