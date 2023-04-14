import cv2
import filters
from managers import WindowManager, CaptureManager


class Cameo(object):

    def __init__(self):
        self._windowManager = WindowManager('Cameo',
                                            self.onKeypress)
        # 不能直接使用 onKeypress()，因为这样会立即调用 onKeypress 方法，
        # 而不是将其作为参数传递给 WindowManager 类的构造函数。
        self._captureManager = CaptureManager(
            cv2.VideoCapture(0), self._windowManager, True)
        # BGR曲线过滤器
        self._curveFilter = filters.BGRPortraCurveFilter()

    def run(self):
        """Run the main loop."""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            # 这里用到了CaptureManager中的frame的getter方法
            frame = self._captureManager.frame

            if frame is not None:
                # 检测边缘
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


if __name__ == "__main__":
    Cameo().run()
