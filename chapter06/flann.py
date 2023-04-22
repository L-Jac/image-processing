import numpy as np
import cv2
from matplotlib import pyplot as plt

img0 = cv2.imread('../images/gauguin_entre_les_lys.jpg', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('../images/gauguin_paintings.png', cv2.IMREAD_GRAYSCALE)

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

# 创建一个空白的掩码，用于在后面的代码中绘制匹配成功的特征点。
# 如果mask_matches=[[0,0],[1,0]]，这意味着有两个匹配的关键点：
# 对于第一个关键点，最优和次优匹配项都是糟糕的；
# 而对于第二个关键点，最佳匹配是好的，次优匹配是糟糕的。
mask_matches = [[0, 0] for i in range(len(matches))]

# 根据劳氏比率测试的结果来更新掩码，以便在后面的代码中绘制匹配成功的特征点。
# 应用乘数为0.7的劳氏比率检验
# enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，
# 同时列出数据和数据下标，一般用在 for 循环当中:
# ['Spring', 'Summer', 'Fall', 'Winter']
#     =>
#        [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        mask_matches[i] = [1, 0]

# 	IMG1 关键点1 IMG2 关键点2 匹配1到2 输出图片 匹配颜色 单点颜色 匹配掩码 标志
img_matches = cv2.drawMatchesKnn(
    img0, kp0, img1, kp1, matches, None,
    matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
    matchesMask=mask_matches, flags=0)

# Show the matches.
cv2.imshow("i", img_matches)
plt.imshow(img_matches)
plt.show()
