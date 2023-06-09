# TODO 建立自定义物体检测器

# 7.2　理解HOG描述符
"""
HOG是一种特征描述符，
因此它与尺度不变特征变换（Scale Invariant Feature Transform，SIFT）、
加速鲁棒特征（Speeded-Up Robust Feature，SURF），
以及ORB（这些已在第6章中介绍过）都属于同一算法家族。
与其他特征描述符一样，HOG能够提供对特征匹配以及对物体检测和识别至关重要的信息类型。
HOG的内部机制非常智能，能把图像划分为若干单元，并针对每个单元计算一组梯度。
每个梯度描述了在给定方向上像素密度的变化。这些梯度共同构成了单元的直方图表示。
在第5章中使用局部二值模式直方图研究人脸识别时，我们遇到过类似的方法。
"""

# 7.3　理解非极大值抑制
"""
非极大值抑制（NMS）的概念听起来可能很简单，即从一组重叠的解中选出一个最好的！
下面是NMS的一个典型实现方法：
（1）构建图像金字塔。
（2）对于物体检测，用滑动窗口方法扫描金字塔的每一层。
    对于每个产生正检测结果（超过某个任意置信度阈值）的窗口，将窗口转换回原始图像尺度。
    将窗口及其置信度添加到正检测结果列表中。
（3）将正检测结果列表按照置信度降序排序，这样最佳检测结果就排在了第一的位置。
（4）对于在正检测结果列表中的每个窗口W，移除与W明显重叠的后续窗口，
    就得到一个满足NMS标准的正检测结果列表。
"""

# 7.5　基于HOG描述符检测人
#   chapter07/detect_people_hog.py

# TODO 7.6 创建并训练物体检测器
#   chapter07/detect_car_bow_svm.py
#   优化：
#       图像缩放是用于生成图像金字塔的。
#       图像金字塔是一种常用的数据结构，它包含了一系列不同尺寸的图像。
#       在物体检测等任务中，图像金字塔可以帮助我们在不同尺度下检测物体。
#       chapter07/detect_car_bow_svm_sliding_window.py
"""
BoW
    BoW是一种技术，可以给一系列文档中的每个单词指定权重或计数，然后用这些计数的向量表示这些文档。
    例子：
        示例文档.png
        构建一个字典，也称为码本（codebook）或词表（vocabulary）
        Codebook_or_Vocabulary.png
        使用8个元素的向量来表示原始文档
        句子向量表.png
        这些向量概念化为文档的直方图表示，或者概念化为用来训练分类器的描述符向量

构建分类器可行思路
    （1）获取图像的一个样本数据集。
    （2）对于数据集中的每一幅图像，（用SIFT、SURF、ORB或者类似的算法）提取描述符。
    （3）向BoW训练器添加描述符向量。
    （4）将描述符聚成k个聚类，这些聚类的中心（质心）是视觉词汇。
    
k均值聚类
    k均值（k-means）聚类是一种量化方法，借此我们可以通过分析大量的向量得到少量的聚类。
    给定一个数据集，k表示该数据集将要划分的聚类数。
    “均值”一词是指数学上的平均数，当直观地表示时，聚类的均值是它的质心或聚类的几何中心点。
"""
