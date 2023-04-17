import numpy


def createMedianMask(disparityMap, validDepthMask, rect=None):
    """返回选择中间图层的蒙版以及阴影。"""
    if rect is not None:
        x, y, w, h = rect
        disparityMap = disparityMap[y:y + h, x:x + w]
        validDepthMask = validDepthMask[y:y + h, x:x + w]
    median = numpy.median(disparityMap)
    return numpy.where((validDepthMask == 0) | (abs(disparityMap - median) < 12), 255, 0).astype(numpy.uint8)
