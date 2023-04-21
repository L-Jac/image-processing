import cv2
from matplotlib import pyplot as plt

# orb.py的优化
img0 = cv2.imread('../images/nasa_logo.png', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('../images/kennedy_space_center.jpg', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()
kp0, des0 = orb.detectAndCompute(img0, None)
kp1, des1 = orb.detectAndCompute(img1, None)

# 使用knnMatch方法来对两张图片的描述符进行KNN匹配，
# 其中k=2表示对于每个描述符，返回最近的两个匹配。
# 使用暴力匹配器对两张图片的描述符进行KNN匹配，以找到最佳匹配对。
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
pairs_of_matches = bf.knnMatch(des0, des1, k=2)

# knnMatch返回列表的列表，每个内部列表至少包含一个匹配项，
# 且不超过k个匹配项，各匹配项从最佳（最短距离）到最差依次排序。
# 根据最佳匹配的距离分值对外部列表进行排序：
pairs_of_matches = sorted(pairs_of_matches, key=lambda x: x[0].distance)

# 画出前25个最佳匹配,以及knnMatch可能与之配对的所有次优匹配。
# 不能使用cv2.drawMatches函数，因为该函数只接受一维匹配列表
# 必须使用cv2.drawMatchesKnn。
img_pairs_of_matches = cv2.drawMatchesKnn(
    img0, kp0, img1, kp1, pairs_of_matches[:25], img1,
    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# 目前为止，我们还没有过滤掉所有糟糕的匹配——实际上，
# 还故意包含了我们认为是糟糕的次优匹配——因此，结果看起来有点乱

# 应用比率检验，把阈值设置为次优匹配距离分值的0.8倍
# 对于每个匹配对，如果长度大于1且第一个匹配的距离小于第二个匹配的距离的0.8倍，
# 则将第一个匹配添加到最终的匹配列表中
matches = []
for pair in pairs_of_matches:
    if len(pair) > 1 and pair[0].distance < 0.8 * pair[1].distance:
        matches.append(pair[0])
# matches = [x[0] for x in pairs_of_matches
#            if len(x) > 1 and x[0].distance < 0.8 * x[1].distance]

# 画出前25个最佳匹配
img_matches = cv2.drawMatches(
    img0, kp0, img1, kp1, matches[:25], img1,
    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

fig, axs = plt.subplots(2, 1)
axs[0].imshow(img_pairs_of_matches)
axs[1].imshow(img_matches)
fig = plt.figure(figsize=(6, 4), dpi=100)
ax = fig.add_subplot(1, 1, 1)
ax.imshow(img1)
plt.show()
