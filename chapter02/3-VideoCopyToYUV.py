import cv2

# 读取视频
videoCapture = cv2.VideoCapture('MyInputVid.avi')
# 获取帧数
fps = videoCapture.get(cv2.CAP_PROP_FPS)
# 获取宽高
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# 将原视频文件写入新视频文件
# VideoWriter_fourcc视频编解码器，I420表示YUV编码，4:2:0色度抽样 兼容性好
videoWriter = cv2.VideoWriter(
    'MyOutputVid.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

success, frame = videoCapture.read()
# 逐帧写入
while success:  # Loop until there are no more frames.
    videoWriter.write(frame)
    success, frame = videoCapture.read()
