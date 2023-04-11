import cv2

clicked = False


# 鼠标交互函数
# 用来检测鼠标左键单击事件。当用户在窗口中单击鼠标左键时，
# onMouse 函数会被调用，并将全局变量 clicked 设置为 True。
# 这样，当 clicked 变量变为 True 时，主循环将退出，窗口将关闭。
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True


# 获取摄像头
cameraCapture = cv2.VideoCapture(0)
# 创建窗口”MyWindow“
cv2.namedWindow('MyWindow')
# 设置鼠标回调函数
cv2.setMouseCallback('MyWindow', onMouse)

print('Showing camera feed. Click window or press any key to stop.')
# 写入视频
success, frame = cameraCapture.read()
# waitkey捕获键盘输入未输入时返回值为-1，1表示等待1毫秒
while success and cv2.waitKey(1) == -1 and not clicked:
    cv2.imshow('MyWindow', frame)
    success, frame = cameraCapture.read()
# 关闭窗口
cv2.destroyWindow('MyWindow')
