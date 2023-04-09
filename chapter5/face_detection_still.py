import cv2


face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
img = cv2.imread('classmates.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.08, 5)
for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
  
cv2.namedWindow('Classmates Detected!')
cv2.imshow('Classmates Detected!', img)
cv2.imwrite('classmates_detected.jpg', img)
cv2.waitKey(0)
