import cv2
from matplotlib import pyplot as plt

# 常规导入，以灰度格式加载两幅图像
img0 = cv2.imread('../images/nasa_logo.png', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('../images/kennedy_space_center.jpg', cv2.IMREAD_GRAYSCALE)

# 创建ORB特征检测器和描述符。
# 检测并计算这两幅图像的关键点和描述符。
orb = cv2.ORB_create()
kp0, des0 = orb.detectAndCompute(img0, None)
kp1, des1 = orb.detectAndCompute(img1, None)

# 执行蛮力匹配。
# 遍历描述符并确定是否匹配，然后计算匹配的质量（距离），
# 并对匹配进行排序，这样就可以在一定程度上显示前n个匹配，
# 它们实际上匹配了两幅图像上的特征。
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des0, des1)


# 按距离对匹配项进行排序。
def get_distance(match):
    return match.distance


matches = sorted(matches, key=get_distance)
# matches = sorted(matches, key=lambda x: x.distance)

# 获取最好的25次结果
img_matches = cv2.drawMatches(
    img0, kp0, img1, kp1, matches[:25], img1,
    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img_matches)
plt.show()
# 这是一个令人失望的结果。实际上，我们可以看到大多数匹配都是假匹配。
# 不幸的是，这很典型。为了改善匹配结果，我们需要应用其他技术来过滤糟糕的匹配。
# 接下来我们将把注意力转向这项任务。
