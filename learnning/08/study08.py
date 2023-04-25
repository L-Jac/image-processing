# TODO 物体跟踪

"""
arXiv（https://arxiv.org/）：这是一个由康奈尔大学图书馆运营的预印本数据库，提供了大量免费的科学论文，包括计算机视觉领域的论文。

Google Scholar（https://scholar.google.com/）：这是一个由谷歌提供的学术搜索引擎，可以搜索到大量免费和付费的学术论文。

Microsoft Academic（https://academic.microsoft.com/）：这是一个由微软提供的学术搜索引擎，可以搜索到大量免费和付费的学术论文。

使用关键词：在搜索框中输入与你要查找的主题相关的关键词，可以帮助你快速找到相关文献。

使用布尔运算符：使用 AND、OR 和 NOT 等布尔运算符可以帮助你更精确地搜索文献。例如，如果你想查找关于计算机视觉和深度学习的文献，可以在搜索框中输入 “计算机视觉 AND 深度学习”。

使用引号：如果你想查找包含特定短语的文献，可以在搜索框中使用引号将短语括起来。例如，如果你想查找包含 “卷积神经网络” 这个短语的文献，可以在搜索框中输入 “卷积神经网络”。

使用高级搜索功能：大多数文献查找网站都提供了高级搜索功能，可以帮助你更精确地搜索文献。例如，你可以指定搜索结果的时间范围、语言、作者等。
"""

# 关于背景差分器
# 不支持detectShadows参数，不支持阴影检测。但是都支持apply方法。
# cv2.bgsegm.createBackgroundSubtractorMOG：基于高斯混合模型的背景/前景分割算法。
# cv2.bgsegm.createBackgroundSubtractorGMG：基于图形模型的背景/前景分割算法。
# cv2.bgsegm.createBackgroundSubtractorCNT：基于计数的背景/前景分割算法。
# cv2.bgsegm.createBackgroundSubtractorGSOC：基于生成序列的背景/前景分割算法。
# cv2.bgsegm.createBackgroundSubtractorLSBP：基于局部自适应灵敏度的背景/前景分割算法。
# (测试用)cv2.bgsegm.createSyntheticSequenceGenerator是一个用于生成合成图像序列的函数。
#       它可以用来创建带有前景物体和背景的合成图像序列，用于'测试'背景减除算法的性能。
#       接受两个输入图像：一个背景图像和一个前景图像。它还接受一个掩码图像，用于指定前景图像中的非零像素。
#       然后，它会在背景图像上随机移动前景物体，生成一系列带有运动前景物体的合成图像。
#       ** = cv2.bgsegm.createSyntheticSequenceGenerator(background, foreground, mask)

# TODO 8.2 基于背景差分检测运动物体
#   1　实现基本背景差分器
#       chapter08/basic_motion_detection.py
#   2　使用MOG背景差分器及其他差分器
#       OpenCV对于MOG背景差分器有两种实现，分别命名为
#       cv2.BackgroundSubtrac-torMOG
#       cv2.BackgroundSubtractorMOG2(增加了对阴影检测的支持)
#       chapter08/mog.py
#       chapter08/gmg.py
#       chapter08/knn.py

# TODO　8.3　利用MeanShift和CamShift跟踪彩色物体
#   HSV模型使用一个不同的三元组通道。色调（hue）是颜色的基调，
#   饱和度（saturation）是颜色的强度，值（value）表示颜色的亮度。
#   cv2.calcHist
#   用于计算颜色直方图，直方图是一种统计图表，它显示了图像中每个像素值出现的频率。
#   参数：learnning/08/cv2_calcHist.png
#   cv2.calcBackProjec
#   用于计算直方图反向投影。直方图反向投影可以用来在一幅图像中寻找与另一幅图像具有相同直方图的区域。
#   参数：learnning/08/cv2_calcBackProject.png
#       MeanShift
"""
MeanShift和CamShift是基于颜色直方图的目标跟踪算法可以在一定条件下有效地跟踪物体
缺点：
    容易受到背景干扰和遮挡的影响，以及不能适应目标的尺度和方向的变化
优化方法：
    使用多种特征来描述目标，如颜色、形状、纹理等，并用融合策略来结合不同特征的反投影权重。
    使用自适应核函数来调整搜索窗口的大小和形状，以适应目标的尺度和方向的变化。
    使用自组织映射（SOM）来对目标进行动态建模和更新，以应对目标外观的变化，并用SOM来生成反投影图像，以提高目标与背景的区分度。
    
"""
# 8.4　使用卡尔曼滤波器寻找运动趋势
#   chapter08/kalman.py

# 8.5 跟踪行人
#   chapter08/track_pedestrians_kalman_meanshift.py

