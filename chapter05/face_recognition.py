#!/usr/bin/env python

# ------------------------------------------------------------------------------------------------
# Note:
# When using the FaceRecognizer interface in combination with Python, please stick to Python 2.
# Some underlying scripts like create_csv will not work in other versions, like Python 3.
# ------------------------------------------------------------------------------------------------

import os
import sys
import cv2
import numpy as np


# 这个函数的主要用途是将给定数组中的值归一化到一个特定范围内。
# 这在数据预处理中非常常见，因为许多机器学习算法对数据的范围和分布都很敏感。
# 通过将数据归一化到一个特定范围内，可以提高算法的性能并避免一些问题。
# 例如，在人脸识别中，我们可能需要将图像的像素值归一化到 0 到 1 之间，以便更好地处理图像数据。这个函数就可以用来完成这个任务。
def normalize(X, low, high, dtype=None):
    """将给定数组 X 中的值归一化到 low 和 high 之间"""
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # 将值归一化到 0 到 1 之间
    X = X - float(minX)
    X = X / float((maxX - minX))
    # 将值缩放到 low 和 high 之间
    X = X * (high - low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)


def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    if filename == ".directory":
                        continue
                    filepath = os.path.join(subject_path, filename)
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    if im is None:
                        print("image " + filepath + " is none")
                    # resize to given size (if given)
                    if sz is not None:
                        im = cv2.resize(im, sz)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError as e:
                    errno, strerror = e.args
                    print("I/O error({0}): {1}".format(errno, strerror))
                except Exception:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise
            c = c + 1
    return [X, y]


if __name__ == "__main__":
    # This is where we write the images, if an output_dir is given
    # in command line:
    out_dir = None
    # You'll need at least a path to your image data, please see
    # the tutorial coming with this source code on how to prepare
    # your image data:
    if len(sys.argv) < 2:
        print("USAGE: facerec_demo.py </path/to/images> [</path/to/store/images/at>]")
        sys.exit()
    # Now read in the image data. This must be a valid path!
    [X, y] = read_images(sys.argv[1])
    # Convert labels to 32bit integers. This is a workaround for 64bit machines,
    # because the labels will truncated else. This will be fixed in code as
    # soon as possible, so Python users don't need to know about this.
    # Thanks to Leo Dirac for reporting:
    y = np.asarray(y, dtype=np.int32)
    # If a out_dir is given, set it:
    if len(sys.argv) == 3:
        out_dir = sys.argv[2]
    # Create the Eigenfaces model. We are going to use the default
    # parameters for this simple example, please read the documentation
    # for thresholding:
    model = cv2.face.createEigenFaceRecognizer()
    # Read
    # Learn the model. Remember our function returns Python lists,
    # so we use np.asarray to turn them into NumPy lists to make
    # the OpenCV wrapper happy:
    model.train(np.asarray(X), np.asarray(y))
    # We now get a prediction from the model! In reality you
    # should always use unseen images for testing your model.
    # But so many people were confused, when I sliced an image
    # off in the C++ version, so I am just using an image we
    # have trained with.
    #
    # model.predict is going to return the predicted label and
    # the associated confidence:
    [p_label, p_confidence] = model.predict(np.asarray(X[0]))
    # Print it:
    print("Predicted label = %d (confidence=%.2f)" % (p_label, p_confidence))
    # Cool! Finally we'll plot the Eigenfaces, because that's
    # what most people read in the papers are keen to see.
    #
    # Just like in C++ you have access to all model internal
    # data, because the cv::FaceRecognizer is a cv::Algorithm.
    #
    # You can see the available parameters with getParams():
    print(model.getParams())
    # Now let's get some data:
    mean = model.getMat("mean")
    eigenvectors = model.getMat("eigenvectors")
    # We'll save the mean, by first normalizing it:
    mean_norm = normalize(mean, 0, 255, dtype=np.uint8)
    mean_resized = mean_norm.reshape(X[0].shape)
    if out_dir is None:
        cv2.imshow("mean", mean_resized)
    else:
        cv2.imwrite("%s/mean.png" % (out_dir), mean_resized)
    # Turn the first (at most) 16 eigenvectors into grayscale
    # images. You could also use cv::normalize here, but sticking
    # to NumPy is much easier for now.
    # Note: eigenvectors are stored by column:
    for i in range(min(len(X), 16)):
        eigenvector_i = eigenvectors[:, i].reshape(X[0].shape)
        eigenvector_i_norm = normalize(eigenvector_i, 0, 255, dtype=np.uint8)
        # Show or save the images:
        if out_dir is None:
            cv2.imshow("%s/eigenface_%d" % (out_dir, i), eigenvector_i_norm)
        else:
            cv2.imwrite("%s/eigenface_%d.png" % (out_dir, i), eigenvector_i_norm)
    # Show the images:
    if out_dir is None:
        cv2.waitKey(0)
