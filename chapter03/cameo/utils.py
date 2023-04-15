import cv2
import numpy
import scipy.interpolate


def createLookupArray(func, length=256):
    """返回对函数整数输入的查找。

    查找值被固定为 [0，长度 - 1]。

    """
    # 将函数值结果直接储存，方便查找，省的调用函数
    if func is None:
        return None
    # 创建一个给定形状的新数组，而不初始化数组中的元素
    lookupArray = numpy.empty(length)
    i = 0
    while i < length:
        func_i = func(i)
        lookupArray[i] = min(max(0, func_i), length - 1)
        i += 1
    return lookupArray


def applyLookupArray(lookupArray, src, dst):
    """使用查找将源映射到目标。"""
    # lookupArray是一个储存函数值结果的数列
    # 查找对应值并映射到新数组
    if lookupArray is None:
        return
    dst[:] = lookupArray[src]


def createCurveFunc(points):
    """返回从控制点派生的函数。"""
    if points is None:
        return None
    numPoints = len(points)
    if numPoints < 2:
        return None
    # points = [(1, 2), (3, 4), (5, 6)]
    # xs, ys = zip(*points)
    # print(xs) # (1, 3, 5)
    # print(ys) # (2, 4, 6)
    xs, ys = zip(*points)
    # 选择不同插值
    if numPoints < 3:
        kind = 'linear'
    elif numPoints < 4:
        kind = 'quadratic'
    else:
        kind = 'cubic'
    # 依靠(xs,ys)结合插值划线，然后根据输入的值找对应点
    return scipy.interpolate.interp1d(xs, ys, kind,
                                      bounds_error=False)


def createCompositeFunc(func0, func1):
    """返回两个函数的组合。"""
    if func0 is None:
        return func1
    if func1 is None:
        return func0
    return lambda x: func0(func1(x))
    # def compositeFunc(x):
    #         return func0(func1(x))
    #
    #     return compositeFunc
