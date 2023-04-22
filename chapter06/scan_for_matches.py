import os

import numpy as np
import cv2

# 常规读图
folder = '../images/tattoos'
query = cv2.imread(os.path.join(folder, 'query.png'),
                   cv2.IMREAD_GRAYSCALE)

# 创建文件、图像、全局描述符
files = []
images = []
descriptors = []
# 将generate_descriptors.py创建好的描述符文件加入描述符列表
for (dirpath, dirnames, filenames) in os.walk(folder):
    files.extend(filenames)
    for f in files:
        if f.endswith('npy') and f != 'query.npy':
            descriptors.append(f)
print(descriptors)

# 创造特征检测器
sift = cv2.SIFT_create()

# 关键字和描述符
query_kp, query_ds = sift.detectAndCompute(query, None)

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



greatest_num_good_matches = 0
prime_suspect = None

print('>> Initiating picture scan...')
for d in descriptors:
    print('--------- analyzing %s for matches ------------' % d)
    matches = flann.knnMatch(
        query_ds, np.load(os.path.join(folder, d)), k=2)
    # 建一个通过了劳氏比率检验的匹配列表
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    num_good_matches = len(good_matches)
    # 用 replace 函数将描述符文件名中的 .npy 后缀替换为空字符串，
    # 然后使用 upper 函数将文件名转换为大写字母。
    # 例如，如果描述符文件名为 image.npy，
    # 那么这段代码将返回 IMAGE，表示图像文件名为 IMAGE.png。
    name = d.replace('.npy', '').upper()
    # 设置最低匹配项
    MIN_NUM_GOOD_MATCHES = 10
    if num_good_matches >= MIN_NUM_GOOD_MATCHES:
        print(f"{name} is a suspect! ({num_good_matches} matches)")
        if num_good_matches > greatest_num_good_matches:
            greatest_num_good_matches = num_good_matches
            prime_suspect = name
    else:
        print(f"{name} is NOT a suspect. ({num_good_matches} matches)")

if prime_suspect is not None:
    print(f"Prime suspect is {prime_suspect}.")
else:
    print('There is no suspect.')
