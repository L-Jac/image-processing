import numpy as np
import cv2
from matplotlib import pyplot as plt

img0 = cv2.imread('../images/tattoos/query.png', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('../images/tattoos/anchor-man.png', cv2.IMREAD_GRAYSCALE)

# 执行 SIFT 特征检测和描述。
sift = cv2.SIFT_create()
# 关键点与描述符
kp0, des0 = sift.detectAndCompute(img0, None)
kp1, des1 = sift.detectAndCompute(img1, None)

# 定义基于 FLANN 的匹配参数。
# FLANN_INDEX_KDTREE = 1 表示使用 KD 树算法来查找最近邻居。
FLANN_INDEX_KDTREE = 1
# 定义了索引参数。
# algorithm=FLANN_INDEX_KDTREE 表示使用 KD 树算法，
# trees=5 表示使用 5 棵树来构建索引。
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# 定义了搜索参数。其中，checks=50 表示在搜索最近邻居时，要检查的节点数量。
# 对每棵树执行50次检查或者遍历。检查次数越多，可以提供的精度也越高，但是计算成本也就更高。
search_params = dict(checks=50)

# 执行基于 FLANN 的匹配。
# FLANN匹配器接受2个参数：indexParams对象和searchParams对象。
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des0, des1, k=2)

# Find all the good matches as per Lowe's ratio test.
# 建一个通过了劳氏比率检验的匹配列表
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 设置最低匹配项
MIN_NUM_GOOD_MATCHES = 10
if len(good_matches) >= MIN_NUM_GOOD_MATCHES:
    # 如果满足这个条件，那么就查找匹配的关键点的二维坐标，并把这些坐标放入浮点坐标对的两个列表中。
    # 一个列表包含查询图像中的关键点坐标，另一个列表包含场景中匹配的关键点坐标：
    src_pts = []
    dst_pts = []
    for m in good_matches:
        src_pt = kp0[m.queryIdx].pt
        dst_pt = kp1[m.trainIdx].pt
        src_pts.append(src_pt)
        dst_pts.append(dst_pt)
    # 形状为 (-1, 1, 2) 的数组第一维的大小由数组中元素的总数确定
    # 例如，如果有 4 个匹配点，那么这个数组的形状就是 (4, 1, 2)
    # 包含了 4 个形状为 (1, 2) 的子数组，每个子数组表示一个匹配点的坐标。
    src_pts = np.float32(src_pts).reshape(-1, 1, 2)
    dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)
    # src_pts = np.float32(
    #     [kp0[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # dst_pts = np.float32(
    #     [kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 寻找单应性：cv2.findHomography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # 用ravel()方法将数组mask拉成一维数组
    mask_matches = mask.ravel().tolist()

    # 执行一个透视转换，取查询图像的矩形角点，并将其投影到场景中，这样就可以画出边界：
    h, w = img0.shape
    src_corners = np.float32(
        [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst_corners = cv2.perspectiveTransform(src_corners, M)
    dst_corners = dst_corners.astype(np.int32)

    # 根据单应性矩阵绘制匹配区域的边界
    num_corners = len(dst_corners)
    for i in range(num_corners):
        x0, y0 = dst_corners[i][0]
        if i == num_corners - 1:
            next_i = 0
        else:
            next_i = i + 1
        x1, y1 = dst_corners[next_i][0]
        # 逐点画线
        cv2.line(img1, (x0, y0), (x1, y1), 255, 3, cv2.LINE_AA)

    # Draw the matches that passed the ratio test.
    img_matches = cv2.drawMatches(
        img0, kp0, img1, kp1, good_matches, None,
        matchColor=(0, 255, 0), singlePointColor=None,
        matchesMask=mask_matches, flags=2)

    # Show the homography and good matches.
    cv2.imshow("i", img_matches)
    plt.imshow(img_matches)
    plt.show()
else:
    # 表示没有找到足够多的好的匹配点。并显示了实际找到的好的匹配点的数量和所需的最小匹配点数量。
    print(f"Not enough matches good were found - {len(good_matches)}/{MIN_NUM_GOOD_MATCHES}")
