import cv2
import numpy as np
import os

# When running in Jupyter, the `non_max_suppression_fast` function should
# already be in the global scope. Otherwise, import it now.
if 'non_max_suppression_fast' not in globals():
    from non_max_suppression import non_max_suppression_fast

"""
if not os.path.isdir('CarData'):
    print('CarData folder not found. Please download and unzip '
          'https://github.com/gcr/arc-evaluator/raw/master/CarData.tar.gz '
          'into the same folder as this script.')
    exit(1)
"""

# 两个训练阶段：
# 一个阶段用于BoW词表，将使用大量图像作为样本；
# 一个阶段用于支持向量机，将使用大量BoW描述符向量作为样本。
# 用于训练 BoW 词表的样本数量
BOW_NUM_TRAINING_SAMPLES_PER_CLASS = 30
# 用于训练SVM 的每个类别的样本数量
SVM_NUM_TRAINING_SAMPLES_PER_CLASS = 110

# 创建 SIFT 特征检测器对象
sift = cv2.SIFT_create()

# 使用 FLANN 算法的参数创建 FLANN 匹配器对象
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# BoW 词表中聚类的数量(k均值聚类)
BOW_NUM_CLUSTERS = 12
# 使用 BOWKMeansTrainer 类创建了一个 BoW KMeans 训练器对象
# 初始化时必须指定聚类数
bow_kmeans_trainer = cv2.BOWKMeansTrainer(BOW_NUM_CLUSTERS)
# 使用 BOWImgDescriptorExtractor 类创建了一个 BoW 图像描述符提取器对象
# 初始化时必须指定描述符提取器和描述符匹配器
bow_extractor = cv2.BOWImgDescriptorExtractor(sift, flann)


def get_pos_and_neg_paths(i):
    # 根据该参数生成两个文件路径，分别表示正样本和负样本的图像文件
    pos_path = f'CarData/TrainImages/pos-{i + 1}.pgm'
    neg_path = f'CarData/TrainImages/neg-{i + 1}.pgm'
    return pos_path, neg_path


def add_sample(path):
    # 常规灰度读图
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # 使用 SIFT 特征检测器检测图像中的关键点和描述符
    keypoints, descriptors = sift.detectAndCompute(img, None)
    # 如果描述符不为空，则将其添加到 BoW KMeans 训练器中。
    if descriptors is not None:
        bow_kmeans_trainer.add(descriptors)


# 添加训练样本:
for i in range(BOW_NUM_TRAINING_SAMPLES_PER_CLASS):
    pos_path, neg_path = get_pos_and_neg_paths(i)
    add_sample(pos_path)
    add_sample(neg_path)

# 调用词表训练器的cluster方法，执行k均值分类并返回词表
voc = bow_kmeans_trainer.cluster()
# 把这个词表分配给BoW描述符提取器
# 在这一阶段，BoW描述符提取器拥有了从高斯差分（Difference of Gaussian，DoG）特征提取BoW描述符所需要的一切。
bow_extractor.setVocabulary(voc)


def extract_bow_descriptors(img):
    # 接受图像并返回由BoW描述符提取器计算的描述符向量。
    # 这涉及图像的DoG特征提取以及基于DoG特征的BoW描述符向量的计算
    features = sift.detect(img)
    return bow_extractor.compute(img, features)


training_data = []
training_labels = []
for i in range(SVM_NUM_TRAINING_SAMPLES_PER_CLASS):
    pos_path, neg_path = get_pos_and_neg_paths(i)
    # 常规灰度读图
    pos_img = cv2.imread(pos_path, cv2.IMREAD_GRAYSCALE)
    # 接受图像并返回由BoW描述符提取器计算的描述符向量
    pos_descriptors = extract_bow_descriptors(pos_img)
    if pos_descriptors is not None:
        training_data.extend(pos_descriptors)
        training_labels.append(1)
    # 常规灰度读图
    neg_img = cv2.imread(neg_path, cv2.IMREAD_GRAYSCALE)
    # 接受图像并返回由BoW描述符提取器计算的描述符向量
    neg_descriptors = extract_bow_descriptors(neg_img)
    if neg_descriptors is not None:
        training_data.extend(neg_descriptors)
        training_labels.append(-1)

# 使用了OpenCV库中的机器学习模块来创建一个支持向量机(SVM)分类器
# 我们创建一个支持向量机，并用之前组建的数据和标签对其进行训练
svm = cv2.ml.SVM_create()
"""
重新加载训练好的SVNM
svm.load('my_svm.xml')
"""
# setType和setC方法都没有返回值，因此不能使用链式调用的方式。
# 即svm = cv2.ml.SVM_create().setType(cv2.ml.SVM_C_SVC).setC(50)
# 设置类型为C-SVC，这是一种常用的SVM类型
svm.setType(cv2.ml.SVM_C_SVC)
# 设置了SVM的C参数为50
svm.setC(50)
# 把训练数据和标签从列表转换为NumPy数组，然后再将它们传递给cv2.ml_SVM的train方法
svm.train(np.array(training_data), cv2.ml.ROW_SAMPLE,
          np.array(training_labels))
# 保存训练好的SVNM
# svm.save('my_svm.xml')


# 生成一系列缩小的图像
def pyramid(img, scale_factor=1.05, min_size=(100, 40),
            max_size=(600, 240)):
    h, w = img.shape
    min_w, min_h = min_size
    max_w, max_h = max_size
    while w >= min_w and h >= min_h:
        if w <= max_w and h <= max_h:
            # 使用yield语句可以让函数返回多个值，并且提高执行效率和减少内存消耗
            # 如果使用return语句，那么函数只能返回一次值
            # 如果将图像添加到一个列表，函数需要等到所有图像都被处理后才能返回结果
            yield img
        # 按比例缩小
        w /= scale_factor
        h /= scale_factor
        # 使用面积插值方法进行缩放可以在缩小图像时保留更多的细节
        # 使用了int函数来将浮点数转换为整数，因为图像的尺寸必须是整数
        img = cv2.resize(img, (int(w), int(h)),
                         interpolation=cv2.INTER_AREA)


# 使用滑动窗口方法在图像上提取多个区域。
# step，控制滑动窗口在图像上移动的步长；
# window_size，指定滑动窗口的尺寸。
def sliding_window(img, step=20, window_size=(100, 40)):
    img_h, img_w = img.shape
    window_w, window_h = window_size
    for y in range(0, img_w, step):
        for x in range(0, img_h, step):
            roi = img[y:y + window_h, x:x + window_w]
            roi_h, roi_w = roi.shape
            if roi_w == window_w and roi_h == window_h:
                yield x, y, roi


# 滑动窗口方法是一种常用的图像处理技术，它用于在图像上提取多个区域。


# 支持向量机分类器的得分阈值
SVM_SCORE_THRESHOLD = 2.2
for test_img_path in ['CarData/TestImages/test-0.pgm',
                      'CarData/TestImages/test-1.pgm',
                      '../images/car.jpg',
                      '../images/haying.jpg',
                      '../images/statue.jpg',
                      '../images/woodcutters.jpg']:
    img = cv2.imread(test_img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 存储检测到的物体的位置和得分
    pos_rects = []
    # 调用pyramid函数对输入图像进行多尺度分析
    for resized in pyramid(gray_img):
        # 调取滑动窗口方法在图像上提取多个区域
        for x, y, roi in sliding_window(resized):
            # 接受图像并返回由BoW描述符提取器计算的描述符向量
            descriptors = extract_bow_descriptors(roi)
            if descriptors is None:
                continue
            # 使用支持向量机分类器对特征/描述符进行分类，predict方法
            # prediction变量是SVM模型的预测结果，一个包含两个元素的元组。
            # 第一个元素是预测类别，第二个元素是一个包含概率估计的数组。
            prediction = svm.predict(descriptors)
            # 如果分类结果为正类，
            if prediction[1][0][0] == 1.0:
                # 则计算分类器的原始得分(置信度)，
                # 也是predict方法但是指定了要返回分类器的原始得分
                raw_prediction = svm.predict(
                    descriptors, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
                # 得分是负数取负
                score = -raw_prediction[1][0][0]
                # 与给定的阈值进行比较
                # 如果得分大于阈值，则将区域的位置和得分添加到pos_rects列表中。
                if score > SVM_SCORE_THRESHOLD:
                    h, w = roi.shape
                    # 原始图像 与 缩小后图像的比值，用于还原尺寸
                    scale = gray_img.shape[0] / float(resized.shape[0])
                    pos_rects.append([int(x * scale),
                                      int(y * scale),
                                      int((x + w) * scale),
                                      int((y + h) * scale),
                                      score])
    # 非极大值抑制算法的重叠阈值
    NMS_OVERLAP_THRESHOLD = 0.4
    # 自己写的NMS函数，在重叠情况下选取得分高的矩形
    pos_rects = non_max_suppression_fast(
        np.array(pos_rects), NMS_OVERLAP_THRESHOLD)
    for x0, y0, x1, y1, score in pos_rects:
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)),
                      (0, 255, 255), 2)
        text = f'{score:.2f}'
        cv2.putText(img, text, (int(x0), int(y0) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow(test_img_path, img)
cv2.waitKey(0)
