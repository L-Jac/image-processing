import cv2
import numpy
import time


# 在Python 2中，class ClassName(object):是一种固定语法，用于定义一个新类，它表示这个类继承自object类。
# object类是Python中所有新式类的基类，它提供了一些基本的方法和属性。
# 在Python 3中，你可以省略(object)，直接使用class ClassName:来定义一个新类，因为在Python 3中所有的类都隐式地继承自object类。
class CaptureManager(object):

    def __init__(self, capture, previewWindowManager=None,
                 shouldMirrorPreview=False):

        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview

        self._capture = capture
        self._channel = 0
        self._enteredFrame = False
        self._frame = None
        self._imageFilename = None
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

        self._startTime = None
        self._framesElapsed = 0
        self._fpsEstimate = None

    @property
    # @property 是一个 Python 内置的装饰器，用于将类方法转换为只读属性
    # channel方法被定义为一个属性。这意味着你可以使用 object.channel 来访问该方法的返回值，
    # 而不是使用 object.channel()。
    def channel(self):
        return self._channel

    @channel.setter
    # @property_name.setter 装饰器来定义一个 setter 方法，用于设置属性的值。
    # 可以使用 object.channel = value 来设置新的通道值。
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            # self._capture.retrieve 方法返回两个值，但只有第二个值（帧）是需要的。
            # 因此，第一个返回值被赋值给 _，表示不关心该返回值。
            # 这种写法可以让代码更简洁、易读。
            _, self._frame = \
                self._capture.retrieve(self._frame, self.channel)
        return self._frame

    @property
    # 如果非空返回self._imageFilename
    def isWritingImage(self):
        return self._imageFilename is not None

    @property
    def isWritingVideo(self):
        return self._videoFilename is not None

    # TODO 实现同步抓取一帧
    def enterFrame(self):

        """Capture the next frame, if any."""

        # But first, check that any previous frame was exited.
        # assert 语句检查 self._enteredFrame 是否为 False。
        # 如果为 True，则抛出一个异常，
        # 异常信息为 'previous enterFrame() had no matching exitFrame()'。
        assert not self._enteredFrame, \
            'previous enterFrame() had no matching exitFrame()'
        # 这种写法与使用 assert 语句的效果相同，但更加明确和易读。
        # if self._enteredFrame:
        #     raise Exception('previous enterFrame() had no matching exitFrame()')
        if self._capture is not None:
            # _capture 是在创建 CaptureManager 类的实例时传入的一个参数。
            # 因此，它可能是一个外部类的实例，具有 grab 方法。
            # 调用grab()方法抓取帧
            self._enteredFrame = self._capture.grab()

    # TODO　实现从当前通道获取图像，估计帧率，显示图像(通过窗口)，完成图像写入文件的请求
    def exitFrame(self):
        """Draw to the window. Write to files. Release the frame."""

        # Check whether any grabbed frame is retrievable.
        # The getter may retrieve and cache the frame.
        # 检查是否有可检索的帧。如果没有，则将 _enteredFrame 属性设置为 False 并返回。
        if self.frame is None:
            self._enteredFrame = False
            return

        # 更新 FPS 估计值和相关变量。如果是第一帧，则记录开始时间。
        # 否则，计算经过的时间并更新 FPS 估计值。
        if self._framesElapsed == 0:
            self._startTime = time.perf_counter()
        else:
            timeElapsed = time.perf_counter() - self._startTime
            self._fpsEstimate = self._framesElapsed / timeElapsed
        self._framesElapsed += 1

        # 如果有预览窗口，则绘制窗口。
        # 如果需要镜像预览，显示翻转后的帧。否则，直接显示原始帧。
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:  # should mirror preview应该镜像预览
                mirroredFrame = numpy.fliplr(self._frame)
                self.previewWindowManager.show(mirroredFrame)
            else:
                self.previewWindowManager.show(self._frame)

        # 如果正在写入图像文件，则使用 cv2.imwrite 方法将帧写入文件，
        # 并将 _imageFilename 属性设置为 None。
        if self.isWritingImage:
            cv2.imwrite(self._imageFilename, self._frame)
            self._imageFilename = None

        # 调用 _writeVideoFrame 方法写入视频帧，方法在下面
        self._writeVideoFrame()

        # 释放帧并将 _enteredFrame 属性设置为 False
        self._frame = None
        self._enteredFrame = False

    # TODO　余下部分都是写入功能
    def writeImage(self, filename):
        """Write the next exited frame to an image file."""
        self._imageFilename = filename

    def startWritingVideo(
            self, filename,
            encoding=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')):
        """Start writing exited frames to a video file."""
        self._videoFilename = filename
        self._videoEncoding = encoding

    def stopWritingVideo(self):
        """Stop writing exited frames to a video file."""
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

    def _writeVideoFrame(self):

        if not self.isWritingVideo:
            return

        if self._videoWriter is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if numpy.isnan(fps) or fps <= 0.0:
                # 如果 FPS 值未知或小于等于 0，则使用 FPS 估计值。
                if self._framesElapsed < 20:
                    # Wait until more frames elapse so that the
                    # estimate is more stable.
                    return
                else:
                    fps = self._fpsEstimate
            size = (int(self._capture.get(
                cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self._capture.get(
                        cv2.CAP_PROP_FRAME_HEIGHT)))
            self._videoWriter = cv2.VideoWriter(
                self._videoFilename, self._videoEncoding,
                fps, size)
        # self._videoWriter 是一个 cv2.VideoWriter 对象，自带write()
        self._videoWriter.write(self._frame)


class WindowManager(object):

    def __init__(self, windowName, keypressCallback=None):
        self.keypressCallback = keypressCallback

        self._windowName = windowName
        self._isWindowCreated = False

    @property
    def isWindowCreated(self):
        return self._isWindowCreated

    def createWindow(self):
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self, frame):
        cv2.imshow(self._windowName, frame)

    def destroyWindow(self):
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def processEvents(self):
        keycode = cv2.waitKey(1)
        if self.keypressCallback is not None and keycode != -1:
            self.keypressCallback(keycode)
