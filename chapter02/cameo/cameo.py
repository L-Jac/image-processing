import cv2
from managers import WindowManager, CaptureManager


class Cameo(object):

    def __init__(self):
        self._windowManager = WindowManager('Cameo',
                                            self.onKeypress)
        # 不能直接使用 onKeypress()，因为这样会立即调用 onKeypress 方法，
        # 而不是将其作为参数传递给 WindowManager 类的构造函数。
        self._captureManager = CaptureManager(
            cv2.VideoCapture(0), self._windowManager, True)

    def run(self):
        """Run the main loop."""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            # 这里用到了CaptureManager中的frame的getter方法
            frame = self._captureManager.frame

            if frame is not None:
                # TODO: Filter the frame (Chapter 3).
                pass

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        """Handle a keypress.

        空格键 -> 拍摄屏幕截图。
        Tab 键 -> 开始/停止录制屏幕。
        Esc 键 -> 退出。

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


# 当你直接运行一个 Python 文件时，它的 __name__ 属性会被设置为 "__main__"。
if __name__ == "__main__":
    Cameo().run()
