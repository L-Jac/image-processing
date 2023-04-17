import cv2
import depth
import filters
from managers import WindowManager, CaptureManager


class Cameo(object):

    def __init__(self):
        self._windowManager = WindowManager('Cameo',
                                            self.onKeypress)
        self._captureManager = CaptureManager(
            cv2.VideoCapture(0), self._windowManager, True)
        self._curveFilter = filters.BGRPortraCurveFilter()

    def run(self):
        """Run the main loop."""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            if frame is not None:
                filters.strokeEdges(frame, frame)
                self._curveFilter.apply(frame, frame)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        """Handle a keypress.

        space  -> Take a screenshot.
        tab    -> Start/stop recording a screencast.
        escape -> Quit.

        """
        if keycode == 32:  # space
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9:  # tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo(
                    'screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27:  # escape
            self._windowManager.destroyWindow()


class CameoDepth(Cameo):

    def __init__(self):
        self._windowManager = WindowManager('Cameo',
                                            self.onKeypress)
        # device = cv2.CAP_OPENNI2 代表微软 Kinect 的设备索引
        device = cv2.CAP_OPENNI2_ASUS  # 代表华硕 Xtion或Occipital结构的索引
        self._captureManager = CaptureManager(
            cv2.VideoCapture(device), self._windowManager, True)
        self._curveFilter = filters.BGRPortraCurveFilter()

    def run(self):
        """Run the main loop."""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            # cv2.CAP_OPENNI_DISPARITY_MAP 8位无符号整数值表示的视差图(一种灰度图像)
            self._captureManager.channel = cv2.CAP_OPENNI_DISPARITY_MAP
            disparityMap = self._captureManager.frame
            # 获取有效深度掩码
            self._captureManager.channel = cv2.CAP_OPENNI_VALID_DEPTH_MASK
            validDepthMask = self._captureManager.frame
            # 表示 BGR 图像的通道
            self._captureManager.channel = cv2.CAP_OPENNI_BGR_IMAGE
            frame = self._captureManager.frame
            if frame is None:
                # 无法捕获 BGR 帧。
                # 尝试捕获红外帧。
                # 表示红外图像的通道
                self._captureManager.channel = cv2.CAP_OPENNI_IR_IMAGE
                frame = self._captureManager.frame

            if frame is not None:

                # 将除中间层以外的所有内容设为黑色。
                # 使用视差图和有效深度掩码创建一个中间层掩码，这个掩码是白色的
                mask = depth.createMedianMask(disparityMap, validDepthMask)
                # 将掩码中值为 0 的像素对应的图像帧中的像素设为黑色
                frame[mask == 0] = 0

                if self._captureManager.channel == cv2.CAP_OPENNI_BGR_IMAGE:
                    # 捕获了 BGR 帧。
                    # 对其应用过滤器。
                    filters.strokeEdges(frame, frame)
                    self._curveFilter.apply(frame, frame)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()


if __name__ == "__main__":
    # Cameo().run() # uncomment for ordinary camera
    CameoDepth().run()  # uncomment for depth camera
