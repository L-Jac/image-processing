import cv2



def is_inside(i, o):
    # i表示内部矩形，o表示外部矩形
    # 如果i在o的内部返回true
    ix, iy, iw, ih = i
    ox, oy, ow, oh = o
    return ix > ox and ix + iw < ox + ow and \
        iy > oy and iy + ih < oy + oh


# 创建cv2.HOGDescriptor实例
hog = cv2.HOGDescriptor()
# 使用OpenCV内置的默认人员检测器
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

img = cv2.imread('../images/haying.jpg')

# detectMultiScale方法，它返回两个列表：
# （1）检测到的物体（在本例中是检测到的人）的矩形框列表。
# （2）检测到的物体的权重或者置信度列表。值越高表示检测结果正确的置信度也就越大。
found_rects, found_weights = hog.detectMultiScale(
        img, winStride=(4, 4), scale=1.02, groupThreshold=1.9)
# detectMultiScale接受几个可选参数，包括：
# ·winStride：这个元组定义了滑动窗口在连续的检测尝试之间移动的x和y距离。
#   HOG可以很好地处理重叠窗口，因此相对于窗口大小，步长可能较小。
#   步长越小，检测次数越多，计算成本也越高。
#   默认步长是让窗口无重叠，也就是与窗口大小相同，即(64,128)，用于默认的人员检测器。
# ·scale：该尺度因子应用于图像金字塔的连续层之间。尺度因子越小，检测次数越多，
#   计算成本也会越高。尺度因子必须大于1.0，默认值是1.5。
# ·finalThreshold：这个值决定检测标准的严格程度。
#   值越小，严格程度越低，检测次数越多。默认值是2.0。

# 检测opencv的版本，4.6为区分
"""
OPENCV_MAJOR_VERSION = int(cv2.__version__.split('.')[0])
OPENCV_MINOR_VERSION = int(cv2.__version__.split('.')[1])

if OPENCV_MAJOR_VERSION >= 5 or (OPENCV_MAJOR_VERSION == 4 and OPENCV_MINOR_VERSION >= 6):
    # 正在使用 OpenCV 4.6 或更高版本。
    found_rects, found_weights = hog.detectMultiScale(
        img, winStride=(4, 4), scale=1.02, groupThreshold=1.9)
else:
    # 正在使用 OpenCV 4.5 或更早版本。
    # groupThreshold 参数曾经被命名为 finalThreshold。
    found_rects, found_weights = hog.detectMultiScale(
        img, winStride=(4, 4), scale=1.02, finalThreshold=1.9)
        """

# 对检测结果进行过滤
found_rects_filtered = []
found_weights_filtered = []
for ri, r in enumerate(found_rects):
    for qi, q in enumerate(found_rects):
        # 去掉嵌套矩形
        if ri != qi and is_inside(r, q):
            break
    else:
        found_rects_filtered.append(r)
        found_weights_filtered.append(found_weights[ri])

for ri, r in enumerate(found_rects_filtered):
    x, y, w, h = r
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    text = f'{found_weights_filtered[ri]:.2f}'
    cv2.putText(img, text, (x, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

cv2.imshow('Women in Hayfield Detected', img)
cv2.imwrite('./women_in_hayfield_detected.png', img)
cv2.waitKey(0)
