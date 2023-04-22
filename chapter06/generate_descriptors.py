import os

import numpy as np
import cv2


def create_descriptors(folder):
    # 接收参数文件夹路径
    # 创造特征检测器
    feature_detector = cv2.SIFT_create()
    files = []
    for (dirpath, dirnames, filenames) in os.walk(folder):
        # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值(用新列表扩展原来的列表)。
        # 元素列表,必须是一个可以迭代的序列,比如数字就不可以
        files.extend(filenames)
    # 遍历指定文件夹中的所有 png 图像文件，并为每个图像文件创建一个描述符文件。
    for f in files:
        create_descriptor(folder, f, feature_detector)


def create_descriptor(folder, image_path, feature_detector):
    # 文件夹路径，图像文件，特征检测器
    # 检查图像文件是否为 png 格式，如果不是，则跳过该文件
    if not image_path.endswith('png'):
        print(f'skipping {image_path}')
        return
    print(f'reading {image_path}')
    img = cv2.imread(os.path.join(folder, image_path),
                     cv2.IMREAD_GRAYSCALE)
    # 关键点与描述符
    keypoints, descriptors = feature_detector.detectAndCompute(img, None)
    # 将描述符保存到一个 npy 文件
    descriptor_file = image_path.replace('png', 'npy')
    np.save(os.path.join(folder, descriptor_file), descriptors)


# 调用函数
folder = '../images/tattoos'
create_descriptors(folder)
