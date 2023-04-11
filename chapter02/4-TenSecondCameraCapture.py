import cv2

# 通过调用cv2.VideoCapture(0)来调用计算机的摄像头，参数0表示计算机的默认摄像头。
cameraCapture = cv2.VideoCapture(0)
# 设定帧数三十帧
fps = 30  # An assumption
# 调取摄像头，获取摄像头宽高
size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# 写入新视频文件
MyOutputVid = "D:/pythonworkspace/image processing/learnning/02/MyOutputVid.avi"
videoWriter = cv2.VideoWriter(
    MyOutputVid, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

success, frame = cameraCapture.read()
# 设置过程持续十秒
numFramesRemaining = 10 * fps - 1  # 10 seconds of frames
# 写入
while numFramesRemaining > 0:
    if frame is not None:
        videoWriter.write(frame)
    success, frame = cameraCapture.read()
    numFramesRemaining -= 1
