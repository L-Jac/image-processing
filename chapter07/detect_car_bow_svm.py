import cv2
import numpy as np
import os

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
# 用于训练 BoW 词表的样本数量(过大会出现过拟合)
BOW_NUM_TRAINING_SAMPLES_PER_CLASS = 10
# 用于训练SVM的每个类别的样本数量
SVM_NUM_TRAINING_SAMPLES_PER_CLASS = 110

# 创建 SIFT 特征检测器对象
sift = cv2.SIFT_create()

# 使用 FLANN 算法的参数创建 FLANN 匹配器对象
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# BoW 词表中聚类的数量(k均值聚类)
BOW_NUM_CLUSTERS = 40
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

# OpenCV提供了名为cv2.ml_SVM的类，代表支持向量机。
# 我们创建一个支持向量机，并用之前组建的数据和标签对其进行训练
svm = cv2.ml.SVM_create()
# 把训练数据和标签从列表转换为NumPy数组，然后再将它们传递给cv2.ml_SVM的train方法
svm.train(np.array(training_data), cv2.ml.ROW_SAMPLE,
          np.array(training_labels))

for test_img_path in ['CarData/TestImages/test-0.pgm',
                      'CarData/TestImages/test-1.pgm',
                      'CarData/car01.png',
                      'CarData/car02.png',
                      'CarData/car03.png',
                      'CarData/car04.png',
                      'CarData/car05.png']:
    # 常规灰度读图
    img = cv2.imread(test_img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 接受图像并返回由BoW描述符提取器计算的描述符向量
    descriptors = extract_bow_descriptors(gray_img)
    # # 使用支持向量机分类器对特征/描述符进行分类
    # prediction变量是SVM模型的预测结果，一个包含两个元素的元组。
    # 第一个元素是预测类别，第二个元素是一个包含概率估计的数组。
    prediction = svm.predict(descriptors)
    if prediction[1][0][0] == 1.0:
        text = 'car'
        color = (0, 255, 0)
    else:
        text = 'not car'
        color = (0, 0, 255)
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color, 2, cv2.LINE_AA)
    cv2.imshow(test_img_path, img)
cv2.waitKey(0)
