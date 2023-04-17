import numpy as np
import cv2
from matplotlib import pyplot as plt

original = cv2.imread('../images/statue_small.jpg')

img = original.copy()
# 创建了一个与图像大小相同的掩码，用于标记前景和背景
# img.shape[:2]表示输入图像的前两个维度，即图像的高度和宽度。
# 在OpenCV中，掩码中的每个元素只能取0、1、2或3这四个值，分别表示像素属于明显的背景、明显的前景、可能的背景和可能的前景。
# 因此，掩码中每个元素只需要用8位就能表示，所以数据类型为np.uint8。
mask = np.zeros(img.shape[:2], np.uint8)

# 创建了两个大小为(1,65)的数组，分别用于存储临时背景模型和临时前景模型
# 这两个数组用于存储临时背景模型和临时前景模型。
# 在执行GrabCut算法时，这两个模型将被更新以更好地拟合背景和前景。
# 由于模型参数是实数，所以这两个数组的数据类型必须是浮点型。
# 在OpenCV中，要求这两个数组的数据类型为np.float64。
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# 定义了一个矩形框，用于指定图像中物体的位置
# 左上角(100,1)，宽421 高378
# (x, y,w, h)
rect = (100, 1, 421, 378)
# img是输入的三通道图像；mask是输入的单通道图像
# rect表示roi区域；bgdModel表示临时背景模型数组；
# fgdModel表示临时前景模型数组；iterCount表示图割算法迭代次数
# 初始化方式为GC_INIT_WITH_RECT表示ROI区域可以被初始化为：
# GC_BGD定义为明显的背景像素0，GC_FGD定义为明显的前景像素1，
# GC_PR_BGD定义为可能的背景像素2，GC_PR_FGD定义为可能的前景像素3
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# 其中mask中等于2或0的元素在新数组中被赋值为0，其余元素被赋值为1。
# 新掩码中，像素属于明显的背景或可能的背景时，对应元素为0；像素属于明显的前景或可能的前景时，对应元素为1。
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
#
img = img * mask2[:, :, np.newaxis]

# 创建了一个1x2的子图网格
plt.subplot(121)
# 将图像从BGR颜色空间转换为RGB颜色空间
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("grabcut")
# plt.xticks([])和plt.yticks([])分别用于隐藏x轴和y轴的刻度。
plt.xticks([])
plt.yticks([])

# plt.subplot(122)将当前子图设置为第二个子图
plt.subplot(122)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title("original")
plt.xticks([])
plt.yticks([])

plt.show()
