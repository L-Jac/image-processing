import cv2
import numpy
import utils


# TODO 实现掩模复制操作

def outlineRect(image, rect, color):
    if rect is None:
        return
    x, y, w, h = rect
    cv2.rectangle(image, (x, y), (x + w, y + h), color)


def copyRect(src, dst, srcRect, dstRect, mask=None,
             interpolation=cv2.INTER_LINEAR):
    """将部分源复制到目标的一部分。"""
    # src 是源图像，dst 是目标图像
    x0, y0, w0, h0 = srcRect
    x1, y1, w1, h1 = dstRect

    # 调整源子矩形的内容大小。
    # 将结果放在目标子矩形中。
    if mask is None:
        # src[y0:y0 + h0, x0:x0 + w0] 表示源图像中的子矩形。
        # 调用 cv2.resize() 函数将这个子矩形调整为目标矩形的大小。
        # 将调整后的结果赋值给 dst[y1:y1 + h1, x1:x1 + w1]，即目标图像中的子矩形。
        # interpolation采用不同插值算法，影响图像缩小质量
        dst[y1:y1 + h1, x1:x1 + w1] = \
            cv2.resize(src[y0:y0 + h0, x0:x0 + w0], (w1, h1),
                       interpolation=interpolation)
    else:
        # 判断是否是灰度图
        if not utils.isGray(src):
            # 将蒙版转换为 3 个通道，如图像。
            mask = mask.repeat(3).reshape(h0, w0, 3)
        # 在应用遮罩的情况下执行复制。
        # 这一行代码表示当掩码中对应位置的值为真时，
        # 返回调整后的源子矩形中对应位置的值；否则返回目标子矩形中对应位置的值。
        dst[y1:y1 + h1, x1:x1 + w1] = \
            numpy.where(cv2.resize(mask, (w1, h1),
                                   interpolation=cv2.INTER_NEAREST),
                        cv2.resize(src[y0:y0 + h0, x0:x0 + w0], (w1, h1),
                                   interpolation=interpolation),
                        dst[y1:y1 + h1, x1:x1 + w1])


def swapRects(src, dst, rects, masks=None, interpolation=cv2.INTER_LINEAR):
    """复制交换了两个或多个子矩形的源。"""

    if dst is not src:
        dst[:] = src

    # 查矩形列表的长度。如果长度小于 2，则直接返回。
    numRects = len(rects)
    if numRects < 2:
        return

    if masks is None:
        masks = [None] * numRects

    # 将最后一个矩形的内容复制到临时存储中。
    x, y, w, h = rects[numRects - 1]
    temp = src[y:y + h, x:x + w].copy()

    # 将每个矩形的内容复制到下一个矩形中。
    i = numRects - 2
    while i >= 0:
        copyRect(src, dst, rects[i], rects[i + 1], masks[i],
                 interpolation)
        i -= 1

    # 将临时存储的内容复制到第一个矩形中。
    copyRect(temp, dst, (0, 0, w, h), rects[0], masks[numRects - 1],
             interpolation)
