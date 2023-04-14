import cv2
import numpy

import utils


def recolorRC(src, dst):
    """该函数用于模拟从BGR颜色空间到RC（红色，青色）颜色空间的转换。
    源图像和目标图像都必须是BGR格式。蓝色和绿色被替换为青色。
    这种效果类似于Technicolor Process 2（用于早期彩色电影）和CGA调色板3（用于早期彩色PC）。

    伪代码如下：
    dst.b = dst.g = 0.5 * (src.b + src.g)
    dst.r = src.r

    """
    # 通道分离
    b, g, r = cv2.split(src)
    # 函数将蓝色和绿色通道的值相加并乘以 0.5，结果存储在蓝色通道中
    # 两个参数0.5是权重，0是偏置
    cv2.addWeighted(b, 0.5, g, 0.5, 0, b)
    # 将蓝色、蓝色和红色通道合并到目标图像中。
    cv2.merge((b, b, r), dst)


def recolorRGV(src, dst):
    """该函数用于模拟从BGR颜色空间到RGV（红色，绿色，值）颜色空间的转换。
    源图像和目标图像都必须是BGR格式。蓝色被去饱和。
    这种效果类似于Technicolor Process 1（用于早期彩色电影）。
    伪代码如下：
    dst.b = min(src.b, src.g, src.r)
    dst.g = src.g
    dst.r = src.r

    """
    b, g, r = cv2.split(src)
    # .min()取最小值
    # 分别计算蓝色、绿色和红色通道中每个像素的最小值，并将结果存储在蓝色通道中。
    cv2.min(b, g, b)
    cv2.min(b, r, b)
    cv2.merge((b, g, r), dst)


def recolorCMV(src, dst):
    """模拟从 BGR 颜色空间到 CMV（青色，品红色，值）颜色空间的转换。
    源图像和目标图像都必须是 BGR 格式。黄色被去饱和。
    这种效果类似于 CGA 调色板 1（用于早期彩色 PC）。
    伪代码：
    dst.b = max(src.b, src.g, src.r)
    dst.g = src.g
    dst.r = src.r

    """
    b, g, r = cv2.split(src)
    cv2.max(b, g, b)
    cv2.max(b, r, b)
    cv2.merge((b, g, r), dst)


def blend(foregroundSrc, backgroundSrc, dst, alphaMask):
    # Calculate the normalized alpha mask.
    maxAlpha = numpy.iinfo(alphaMask.dtype).max
    normalizedAlphaMask = (1.0 / maxAlpha) * alphaMask

    # Calculate the normalized inverse alpha mask.
    normalizedInverseAlphaMask = \
        numpy.ones_like(normalizedAlphaMask)
    normalizedInverseAlphaMask[:] = \
        normalizedInverseAlphaMask - normalizedAlphaMask

    # Split the channels from the sources.
    foregroundChannels = cv2.split(foregroundSrc)
    backgroundChannels = cv2.split(backgroundSrc)

    # Blend each channel.
    numChannels = len(foregroundChannels)
    i = 0
    while i < numChannels:
        backgroundChannels[i][:] = \
            normalizedAlphaMask * foregroundChannels[i] + \
            normalizedInverseAlphaMask * backgroundChannels[i]
        i += 1

    # Merge the blended channels into the destination.
    cv2.merge(backgroundChannels, dst)


def strokeEdges(src, dst, blurKsize=7, edgeKsize=5):
    # blurKsize 参数表示中值模糊内核的大小。
    # 内核越大，模糊程度越高，去噪效果也越好。过大可能导致图像过度模糊。
    # 但是，内核大小必须是奇数且大于等于 3。
    # 如果 blurKsize 小于 3，则不进行中值模糊处理。
    if blurKsize >= 3:
        # cv2.medianBlur用于消除噪声
        # 中值模糊是一种非线性滤波器，
        # 这种方法对于去除椒盐噪声等非高斯噪声效果较好，但计算量相对较大。
        blurredSrc = cv2.medianBlur(src, blurKsize)
        # 转成灰度图以备边缘搜索
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # 拉普拉斯变换进行边缘搜索
    # cv2.Laplacian 函数接受四个参数：输入图像、输出图像的深度、输出图像和内核大小。
    # cv2.CV_8U 表示 8 位无符号整数，cv2.CV_16S 表示 16 位有符号整数，
    # cv2.CV_32F 表示 32 位浮点数等
    # 此时边缘偏暗
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKsize)
    # 计算归一化的反向 alpha 值。
    # 这一步通过将灰度图像的每个像素值从 255 减去，然后乘以 1.0 / 255 来实现。
    # 这样，边缘处的像素值较低，非边缘处的像素值较高。
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)

    channels = cv2.split(src)
    for channel in channels:
        # channel 变量表示源图像的一个通道，它是一个二维数组。
        # channel[:] 表示对这个二维数组中的所有元素进行操作。
        # 此时边缘偏亮
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, dst)


class VFuncFilter(object):
    """A filter that applies a function to V (or all of BGR)."""

    def __init__(self, vFunc=None, dtype=numpy.uint8):
        length = numpy.iinfo(dtype).max + 1
        self._vLookupArray = utils.createLookupArray(vFunc, length)

    def apply(self, src, dst):
        """Apply the filter with a BGR or gray source/destination."""
        srcFlatView = numpy.ravel(src)
        dstFlatView = numpy.ravel(dst)
        utils.applyLookupArray(self._vLookupArray, srcFlatView,
                               dstFlatView)


class VCurveFilter(VFuncFilter):
    """A filter that applies a curve to V (or all of BGR)."""

    def __init__(self, vPoints, dtype=numpy.uint8):
        VFuncFilter.__init__(self, utils.createCurveFunc(vPoints),
                             dtype)


class BGRFuncFilter(object):
    """一个对每个BGR应用不同函数的滤镜。"""

    def __init__(self, vFunc=None, bFunc=None, gFunc=None,
                 rFunc=None, dtype=numpy.uint8):
        # 如果dtype是numpy.uint8，则最大值为255。
        # length = numpy.iinfo(dtype).max + 1计算给定整数类型的最大值加1的结果
        length = numpy.iinfo(dtype).max + 1
        # 见 utils.py
        self._bLookupArray = utils.createLookupArray(
            utils.createCompositeFunc(bFunc, vFunc), length)
        self._gLookupArray = utils.createLookupArray(
            utils.createCompositeFunc(gFunc, vFunc), length)
        self._rLookupArray = utils.createLookupArray(
            utils.createCompositeFunc(rFunc, vFunc), length)

    def apply(self, src, dst):
        """使用BGR源/目标应用滤镜。"""
        b, g, r = cv2.split(src)
        utils.applyLookupArray(self._bLookupArray, b, b)
        utils.applyLookupArray(self._gLookupArray, g, g)
        utils.applyLookupArray(self._rLookupArray, r, r)
        cv2.merge([b, g, r], dst)


# BGR曲线过滤器
class BGRCurveFilter(BGRFuncFilter):
    """对每个 BGR 应用不同曲线的筛选器。"""

    def __init__(self, vPoints=None, bPoints=None,
                 gPoints=None, rPoints=None, dtype=numpy.uint8):
        BGRFuncFilter.__init__(self,
                               utils.createCurveFunc(vPoints),
                               utils.createCurveFunc(bPoints),
                               utils.createCurveFunc(gPoints),
                               utils.createCurveFunc(rPoints), dtype)


class BGRCrossProcessCurveFilter(BGRCurveFilter):
    """A filter that applies cross-process-like curves to BGR."""

    def __init__(self, dtype=numpy.uint8):
        BGRCurveFilter.__init__(
            self,
            bPoints=[(0, 20), (255, 235)],
            gPoints=[(0, 0), (56, 39), (208, 226), (255, 255)],
            rPoints=[(0, 0), (56, 22), (211, 255), (255, 255)],
            dtype=dtype)


class BGRPortraCurveFilter(BGRCurveFilter):
    """A filter that applies Portra-like curves to BGR."""

    def __init__(self, dtype=numpy.uint8):
        BGRCurveFilter.__init__(
            self,
            vPoints=[(0, 0), (23, 20), (157, 173), (255, 255)],
            bPoints=[(0, 0), (41, 46), (231, 228), (255, 255)],
            gPoints=[(0, 0), (52, 47), (189, 196), (255, 255)],
            rPoints=[(0, 0), (69, 69), (213, 218), (255, 255)],
            dtype=dtype)


class BGRProviaCurveFilter(BGRCurveFilter):
    """A filter that applies Provia-like curves to BGR."""

    def __init__(self, dtype=numpy.uint8):
        BGRCurveFilter.__init__(
            self,
            bPoints=[(0, 0), (35, 25), (205, 227), (255, 255)],
            gPoints=[(0, 0), (27, 21), (196, 207), (255, 255)],
            rPoints=[(0, 0), (59, 54), (202, 210), (255, 255)],
            dtype=dtype)


class BGRVelviaCurveFilter(BGRCurveFilter):
    """一个滤镜，它将类似Velvia的曲线应用于BGR"""
    # Velvia是日本富士胶片公司生产的一种日光型彩色反转胶片的品牌。
    # 它以其极高的色彩饱和度和图像质量而闻名。
    # Velvia曲线是指类似于Velvia胶片所呈现的色彩饱和度和图像质量的曲线。
    def __init__(self, dtype=numpy.uint8):
        BGRCurveFilter.__init__(
            self,
            vPoints=[(0, 0), (128, 118), (221, 215), (255, 255)],
            bPoints=[(0, 0), (25, 21), (122, 153), (165, 206), (255, 255)],
            gPoints=[(0, 0), (25, 21), (95, 102), (181, 208), (255, 255)],
            rPoints=[(0, 0), (41, 28), (183, 209), (255, 255)],
            dtype=dtype)


class VConvolutionFilter(object):
    """将卷积应用于 V（或所有 BGR）的筛选器。"""

    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        """使用 BGR 或灰色源/目标应用过滤器。"""
        # src是原图像，ddepth是目标图像的所需深度，kernel是卷积核。
        # -1表示目标图像和原图具有相同的深度
        # dst：目标图像，与原图像尺寸和通过数相同
        cv2.filter2D(src, -1, self._kernel, dst)


class BlurFilter(VConvolutionFilter):
    """半径为 2 像素的模糊滤镜。"""
    # 简单的模糊滤镜

    def __init__(self):
        kernel = numpy.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self, kernel)


class SharpenFilter(VConvolutionFilter):
    """半径为 1 像素的锐化滤镜。"""
    # 如果把权值9改为8就得到了边缘检测核，边缘变白
    # 见FindEdgesFilter
    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class FindEdgesFilter(VConvolutionFilter):
    """半径为 1 像素的边缘查找滤镜。"""

    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
                              [-1, 8, -1],
                              [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class EmbossFilter(VConvolutionFilter):
    """半径为 1 像素的浮雕滤镜。"""

    def __init__(self):
        kernel = numpy.array([[-2, -1, 0],
                              [-1, 1, 1],
                              [0, 1, 2]])
        VConvolutionFilter.__init__(self, kernel)
