#!/usr/bin/env python


"""
See the project's GitHub page at:
https://github.com/JoeHowse/VisualizingTheInvisible
"""

import math  # math模块进行三角计算
import timeit  # timeit模块进行精确的时间测量
import cv2
import numpy


# 灰度转换的辅助函数
def convert_bgr_to_gray(src, dst=None):
    # 使用简单的每个像素的B、G、R值的（非加权）平均值执行灰度转换。
    # 这种方法的计算成本很低（这在实时跟踪中是可取的）
    weight = 1.0 / 3.0
    m = numpy.array([[weight, weight, weight]], numpy.float32)
    return cv2.transform(src, m, dst)


# 将一组二维点映射到三维平面上
def map_2D_points_onto_3D_plane(points_2D, image_size,
                                image_real_height):
    w, h = image_size
    # 计算图像的缩放比例
    image_scale = image_real_height / h

    points_3D = []
    for point_2D in points_2D:
        x, y = point_2D
        # 计算出三维平面上的点的坐标。
        # x 坐标等于图像缩放比例乘以（x 减去图像宽度的一半），
        # y 坐标等于图像缩放比例乘以（y 减去图像高度的一半），
        # z 坐标为 0。
        point_3D = (image_scale * (x - 0.5 * w),
                    image_scale * (y - 0.5 * h),
                    0.0)
        points_3D.append(point_3D)
    return numpy.array(points_3D, numpy.float32)


def map_vertices_to_plane(image_size, image_real_height):
    w, h = image_size

    # 计算出图像的四个顶点的坐标，并将它们存储在 vertices_2D 列表
    vertices_2D = [(0, 0), (w, 0), (w, h), (0, h)]
    #  vertex_indices_by_face 的列表，表示每个面由哪些顶点组成。
    #  在这个例子中，只有一个面，它由四个顶点组成。
    vertex_indices_by_face = [[0, 1, 2, 3]]

    # 调用 map_2D_points_onto_3D_plane 函数将二维顶点映射到三维平面上，
    # 并将结果存储在 vertices_3D 列表(三维顶点的列表）
    vertices_3D = map_2D_points_onto_3D_plane(
        vertices_2D, image_size, image_real_height)
    return vertices_3D, vertex_indices_by_face


class ImageTrackingDemo:

    # 初始化器将为参考图像设置采集设备、摄像头矩阵、卡尔曼滤波器以及2D和3D关键点。
    def __init__(self, capture, diagonal_fov_degrees=70.0,
                 target_fps=25.0,
                 reference_image_path='reference_image.png',
                 reference_image_real_height=1.0):

        # 假设镜头没有经历任何畸变
        self._distortion_coefficients = None

        # 罗德里格斯旋转向量，用于表示三维旋转。
        self._rodrigues_rotation_vector = numpy.array(
            [[0.0], [0.0], [1.0]], numpy.float32)
        # 欧拉旋转向量，也用于表示三维旋转。
        self._euler_rotation_vector = numpy.zeros((3, 1), numpy.float32)  # Radians
        # 旋转矩阵，用于将三维坐标旋转到新的坐标系中。
        self._rotation_matrix = numpy.zeros((3, 3), numpy.float64)
        # 平移向量，用于将三维坐标平移到新的位置。
        self._translation_vector = numpy.zeros((3, 1), numpy.float32)

        # 卡尔曼滤波器，用于估计系统的状态。
        self._kalman = cv2.KalmanFilter(18, 6)
        # numpy.identity()创造单位矩阵E
        # 过程噪声协方差矩阵，表示系统过程中的不确定性。
        self._kalman.processNoiseCov = numpy.identity(
            18, numpy.float32) * 1e-5
        # 测量噪声协方差矩阵，表示测量中的不确定性。
        self._kalman.measurementNoiseCov = numpy.identity(
            6, numpy.float32) * 1e-2
        # 后验误差协方差矩阵，表示估计状态的不确定性。
        self._kalman.errorCovPost = numpy.identity(
            18, numpy.float32)

        # 该卡尔曼滤波器将基于6个输入变量（或测量值）跟踪18个输出变量（或预测）
        # 输入变量是6DOF跟踪结果的元素：tx、ty、tz、rx、ry和rz。
        # 输出变量是经过稳定的6DOF跟踪结果的元素，和一阶导数(速度)和二阶导数(加速度)
        # 顺序如下：
        # tx,ty,tz,tx',ty',tz',tx",ty",tz",rx,ry,rz,rx',ry',rz',rx",ry",rz"
        self._kalman.measurementMatrix = numpy.array(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            numpy.float32)

        # 根据目标帧率(25)来计算转移矩阵中的元素值
        self._init_kalman_transition_matrix(target_fps)

        # 显示是否成功跟踪了上一帧中的物体
        self._was_tracking = False

        # 获得图像尺寸
        self._reference_image_real_height = reference_image_real_height
        # 我们将3D轴箭头的长度定义为打印图像高度的一半
        reference_axis_length = 0.5 * reference_image_real_height

        # -----------------------------------------------------------------------------
        # OpenCV 的坐标系有非标准的轴方向：
        # +X: 物体的左手方向，或者在物体的正面视图中是观察者右手方向。
        # +Y: 表示向下。
        # +Z: 是物体的后向方向，或者在物体正面视图中是观察者的前向方向
        #
        # 我们必须对上述所有方向取反，以获得下述标准的右手坐标系，
        # 就像OpenGL等许多3D图形框架中所使用的那样：
        # +X: 是物体的右手方向，或者在物体的正面视图中是观察者的左手方向。
        # +Y: 表示向上。
        # +Z: 是物体的前向方向，或者在物体的正面视图中是观察者的后向方向。
        #
        # -----------------------------------------------------------------------------

        # 定义相对于打印图像中心[0.0,0.0,0.0]的轴箭头的顶点：
        # 图像中心，指向x轴的负方向，指向y轴负方向，指向z轴负方向
        self._reference_axis_points_3D = numpy.array(
            [[0.0, 0.0, 0.0],
             [-reference_axis_length, 0.0, 0.0],
             [0.0, -reference_axis_length, 0.0],
             [0.0, 0.0, -reference_axis_length]], numpy.float32)

        # 使用3个数组来保存3种类型的图像：
        # BGR视频帧（用于绘制AR图）
        self._bgr_image = None
        # 帧的灰度版本（用于关键点匹配）
        self._gray_image = None
        # 掩模（绘制被跟踪物体的轮廓）
        self._mask = None

        self._capture = capture
        success, trial_image = capture.read()
        if success:
            # 前两个元素
            h, w = trial_image.shape[:2]
        else:
            # 常规获取法
            w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._image_size = (w, h)

        diagonal_image_size = (w ** 2.0 + h ** 2.0) ** 0.5  # √(w^2+h^2)
        diagonal_fov_radians = diagonal_fov_degrees * math.pi / 180.0  # 算角度θ
        # 计算焦距长度 √(w^2+h^2) / (2 * tan(θ/2))
        focal_length = 0.5 * diagonal_image_size / math.tan(0.5 * diagonal_fov_radians)
        # 摄像头矩阵
        self._camera_matrix = numpy.array(
            [[focal_length, 0.0, 0.5 * w],
             [0.0, focal_length, 0.5 * h],
             [0.0, 0.0, 1.0]], numpy.float32)

        """
        调整参考图像的大小时，我们选择将其设置为摄像头帧高的两倍。
        具体倍数可随意，可是，总体思路是想要用覆盖各种放大倍数的图像金字塔执行关键点检测及描述。
        金字塔底部（也就是调整后的参考图像）应该比摄像头帧大，这样当目标物体离摄像头很近不能完全放入摄像头帧时，也可以用合适的尺度匹配关键点。
        相反，金字塔的顶部应该比摄像头帧小，这样即使当目标物体太远而不能填满整个摄像头帧时，也可以以合适的尺度匹配关键点。
        """

        # 读图
        bgr_reference_image = cv2.imread(
            reference_image_path, cv2.IMREAD_COLOR)
        # 获取高宽
        reference_image_h, reference_image_w = bgr_reference_image.shape[:2]
        # 确定缩放因子
        reference_image_resize_factor = (2.0 * h) / reference_image_h
        # 参考图像将被缩放到新的大小，以便适应窗口的高度。
        # (0, 0)：表示输出图像的大小。
        #   由于这里指定为 (0, 0)，因此输出图像的大小将根据缩放因子自动计算。
        # None：表示不使用掩码。
        # reference_image_resize_factor：表示水平方向上的缩放因子。
        # reference_image_resize_factor：表示垂直方向上的缩放因子。
        # cv2.INTER_CUBIC：表示使用三次插值算法进行缩放。
        bgr_reference_image = cv2.resize(
            bgr_reference_image, (0, 0), None,
            reference_image_resize_factor,
            reference_image_resize_factor, cv2.INTER_CUBIC)
        # 调用灰度转换函数
        gray_reference_image = convert_bgr_to_gray(bgr_reference_image)
        # 创建空的掩模，numpy.empty_like 函数返回一个未初始化的数组，它的元素值是未定义的。
        reference_mask = numpy.empty_like(gray_reference_image)

        # 描述符覆盖的直径是31个像素
        patchSize = 31
        # nfeatures=250：表示要检测的特征点的最大数量。
        # scaleFactor=1.2：表示图像金字塔的比例因子。
        # nlevels=16：表示图像金字塔的层数。
        # edgeThreshold=patchSize：表示边缘阈值，用于过滤掉靠近边缘的特征点。
        # patchSize=patchSize：表示特征点描述符的大小。
        self._feature_detector = cv2.ORB_create(
            nfeatures=250, scaleFactor=1.2, nlevels=16,
            edgeThreshold=patchSize, patchSize=patchSize)

        # 将参考图像分成6x6个区域，然后在每个区域内使用掩模来检测关键点和计算描述符。
        # 这种划分模式有助于确保在每个区域(250个关键点)都有一些关键点和描述符，
        # 因此即使在给定帧中物体的大部分不可见的情况下，也可以潜在地匹配关键点并跟踪物体。
        reference_keypoints = []
        # 创建了一个空的numpy数组，用于存储参考图像的描述符。有0行和32列
        # 没有任何行，实际上不包含任何数据。常用作占位符，可以在后续的计算中添加数据。
        self._reference_descriptors = numpy.empty(
            (0, 32), numpy.uint8)
        num_segments_y = 6
        num_segments_x = 6
        for segment_y, segment_x in numpy.ndindex(
                (num_segments_y, num_segments_x)):
            # 描述符patchSize覆盖的直径是31个像素
            y0 = reference_image_h * segment_y // num_segments_y - patchSize
            x0 = reference_image_w * segment_x // num_segments_x - patchSize
            y1 = reference_image_h * (segment_y + 1) // num_segments_y + patchSize
            x1 = reference_image_w * (segment_x + 1) // num_segments_x + patchSize
            reference_mask.fill(0)
            cv2.rectangle(
                reference_mask, (x0, y0), (x1, y1), 255, cv2.FILLED)
            # 用ORB特征检测器检测关键点和计算描述符
            more_reference_keypoints, more_reference_descriptors = \
                self._feature_detector.detectAndCompute(
                    gray_reference_image, reference_mask)
            if more_reference_descriptors is None:
                # 这个区域没描述符就跳过
                continue
            reference_keypoints += more_reference_keypoints
            self._reference_descriptors = numpy.vstack(
                (self._reference_descriptors,
                 more_reference_descriptors))

        # 画关键点
        cv2.drawKeypoints(
            gray_reference_image, reference_keypoints,
            bgr_reference_image,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # 保存图像
        ext_i = reference_image_path.rfind('.')
        reference_image_keypoints_path = \
            reference_image_path[:ext_i] + '_keypoints' + \
            reference_image_path[ext_i:]
        cv2.imwrite(
            reference_image_keypoints_path, bgr_reference_image)

        # flann匹配器索引的参数
        # 使用局部敏感哈希（LSH）索引
        FLANN_INDEX_LSH = 6
        # table_number：LSH索引中哈希表的数量。
        # key_size：每个哈希表中桶的数量。
        # multi_probe_level：探测级别，用于控制搜索时检查的桶的数量。
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6, key_size=12,
                            multi_probe_level=1)
        # 空 表示使用默认的搜索参数。
        search_params = dict()
        # 执行flann
        self._descriptor_matcher = cv2.FlannBasedMatcher(
            index_params, search_params)
        self._descriptor_matcher.add([self._reference_descriptors])

        # pt属性返回一个包含两个元素的元组，表示关键点的x和y坐标。
        reference_points_2D = [keypoint.pt
                               for keypoint in reference_keypoints]
        # [::-1]反转图像的形状 (height, width)=>>(width, height)
        self._reference_points_3D = map_2D_points_onto_3D_plane(
            reference_points_2D, gray_reference_image.shape[::-1],
            reference_image_real_height)

        # 调用map_vertices_to_plane获得参考图像的3D顶点和每个面对应的顶点索引
        (self._reference_vertices_3D,
         self._reference_vertex_indices_by_face) = \
            map_vertices_to_plane(
                gray_reference_image.shape[::-1],
                reference_image_real_height)

    def run(self):

        # 总帧数
        num_images_captured = 0
        start_time = timeit.default_timer()

        while cv2.waitKey(1) != 27:  # Escape
            success, self._bgr_image = self._capture.read(
                self._bgr_image)
            if success:
                num_images_captured += 1
                self._track_object()
                cv2.imshow('Image Tracking', self._bgr_image)
            delta_time = timeit.default_timer() - start_time
            if delta_time > 0.0:
                # 计算帧数
                fps = num_images_captured / delta_time
                self._init_kalman_transition_matrix(fps)

    def _track_object(self):

        self._gray_image = convert_bgr_to_gray(
            self._bgr_image, self._gray_image)

        if self._mask is None:
            self._mask = numpy.full_like(self._gray_image, 255)

        keypoints, descriptors = \
            self._feature_detector.detectAndCompute(
                self._gray_image, self._mask)

        # Find the 2 best matches for each descriptor.
        matches = self._descriptor_matcher.knnMatch(descriptors, 2)

        # Filter the matches based on the distance ratio test.
        good_matches = [
            match[0] for match in matches
            if len(match) > 1 and \
               match[0].distance < 0.8 * match[1].distance
        ]

        # Select the good keypoints and draw them in red.
        good_keypoints = [keypoints[match.queryIdx]
                          for match in good_matches]
        cv2.drawKeypoints(self._gray_image, good_keypoints,
                          self._bgr_image, (0, 0, 255))

        min_good_matches_to_start_tracking = 8
        min_good_matches_to_continue_tracking = 6
        num_good_matches = len(good_matches)

        if num_good_matches < min_good_matches_to_continue_tracking:
            self._was_tracking = False
            self._mask.fill(255)

        elif num_good_matches >= \
                min_good_matches_to_start_tracking or \
                self._was_tracking:

            # Select the 2D coordinates of the good matches.
            # They must be in an array of shape (N, 1, 2).
            good_points_2D = numpy.array(
                [[keypoint.pt] for keypoint in good_keypoints],
                numpy.float32)

            # Select the 3D coordinates of the good matches.
            # They must be in an array of shape (N, 1, 3).
            good_points_3D = numpy.array(
                [[self._reference_points_3D[match.trainIdx]]
                 for match in good_matches],
                numpy.float32)

            # Solve for the pose and find the inlier indices.
            (success, rodrigues_rotation_vector_temp,
             translation_vector_temp, inlier_indices) = \
                cv2.solvePnPRansac(good_points_3D, good_points_2D,
                                   self._camera_matrix,
                                   self._distortion_coefficients,
                                   None, None,
                                   useExtrinsicGuess=False,
                                   iterationsCount=100,
                                   reprojectionError=8.0,
                                   confidence=0.99,
                                   flags=cv2.SOLVEPNP_ITERATIVE)

            if success:

                self._translation_vector[:] = translation_vector_temp
                self._rodrigues_rotation_vector[:] = \
                    rodrigues_rotation_vector_temp
                self._convert_rodrigues_to_euler()

                if not self._was_tracking:
                    self._init_kalman_state_matrices()
                    self._was_tracking = True

                self._apply_kalman()

                # Select the inlier keypoints.
                inlier_keypoints = [good_keypoints[i]
                                    for i in inlier_indices.flat]

                # Draw the inlier keypoints in green.
                cv2.drawKeypoints(self._bgr_image, inlier_keypoints,
                                  self._bgr_image, (0, 255, 0))

                # Draw the axes of the tracked object.
                self._draw_object_axes()

                # Make and draw a mask around the tracked object.
                self._make_and_draw_object_mask()

    def _init_kalman_transition_matrix(self, fps):

        if fps <= 0.0:
            return

        # Velocity transition rate
        vel = 1.0 / fps

        # Acceleration transition rate
        acc = 0.5 * (vel ** 2.0)

        self._kalman.transitionMatrix = numpy.array(
            [[1.0, 0.0, 0.0, vel, 0.0, 0.0, acc, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0, acc, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0, acc,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              1.0, 0.0, 0.0, vel, 0.0, 0.0, acc, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0, acc, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0, acc],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
            numpy.float32)

    def _init_kalman_state_matrices(self):

        t_x, t_y, t_z = self._translation_vector.flat
        pitch, yaw, roll = self._euler_rotation_vector.flat

        self._kalman.statePre = numpy.array(
            [[t_x], [t_y], [t_z],
             [0.0], [0.0], [0.0],
             [0.0], [0.0], [0.0],
             [pitch], [yaw], [roll],
             [0.0], [0.0], [0.0],
             [0.0], [0.0], [0.0]], numpy.float32)
        self._kalman.statePost = numpy.array(
            [[t_x], [t_y], [t_z],
             [0.0], [0.0], [0.0],
             [0.0], [0.0], [0.0],
             [pitch], [yaw], [roll],
             [0.0], [0.0], [0.0],
             [0.0], [0.0], [0.0]], numpy.float32)

    def _apply_kalman(self):

        self._kalman.predict()

        t_x, t_y, t_z = self._translation_vector.flat
        pitch, yaw, roll = self._euler_rotation_vector.flat

        estimate = self._kalman.correct(numpy.array(
            [[t_x], [t_y], [t_z],
             [pitch], [yaw], [roll]], numpy.float32))

        translation_estimate = estimate[0:3]
        euler_rotation_estimate = estimate[9:12]

        self._translation_vector[:] = translation_estimate

        angular_delta = cv2.norm(self._euler_rotation_vector,
                                 euler_rotation_estimate, cv2.NORM_L2)

        MAX_ANGULAR_DELTA = 30.0 * math.pi / 180.0
        if angular_delta > MAX_ANGULAR_DELTA:
            # The rotational motion stabilization seems to be drifting
            # too far, probably due to an Euler angle singularity.

            # Reset the rotational motion stabilization.
            # Let the translational motion stabilization continue as-is.

            self._kalman.statePre[9] = pitch
            self._kalman.statePre[10] = yaw
            self._kalman.statePre[11] = roll
            self._kalman.statePre[12:18] = 0.0

            self._kalman.statePost[9] = pitch
            self._kalman.statePost[10] = yaw
            self._kalman.statePost[11] = roll
            self._kalman.statePost[12:18] = 0.0
        else:
            self._euler_rotation_vector[:] = euler_rotation_estimate
            self._convert_euler_to_rodrigues()

    def _convert_rodrigues_to_euler(self):

        self._rotation_matrix, jacobian = cv2.Rodrigues(
            self._rodrigues_rotation_vector, self._rotation_matrix)

        m00 = self._rotation_matrix[0, 0]
        m02 = self._rotation_matrix[0, 2]
        m10 = self._rotation_matrix[1, 0]
        m11 = self._rotation_matrix[1, 1]
        m12 = self._rotation_matrix[1, 2]
        m20 = self._rotation_matrix[2, 0]
        m22 = self._rotation_matrix[2, 2]

        # Convert to Euler angles using the yaw-pitch-roll
        # Tait-Bryan convention.
        if m10 > 0.998:
            # The rotation is near the "vertical climb" singularity.
            pitch = 0.5 * math.pi
            yaw = math.atan2(m02, m22)
            roll = 0.0
        elif m10 < -0.998:
            # The rotation is near the "nose dive" singularity.
            pitch = -0.5 * math.pi
            yaw = math.atan2(m02, m22)
            roll = 0.0
        else:
            pitch = math.asin(m10)
            yaw = math.atan2(-m20, m00)
            roll = math.atan2(-m12, m11)

        self._euler_rotation_vector[0] = pitch
        self._euler_rotation_vector[1] = yaw
        self._euler_rotation_vector[2] = roll

    def _convert_euler_to_rodrigues(self):

        pitch = self._euler_rotation_vector[0]
        yaw = self._euler_rotation_vector[1]
        roll = self._euler_rotation_vector[2]

        cyaw = math.cos(yaw)
        syaw = math.sin(yaw)
        cpitch = math.cos(pitch)
        spitch = math.sin(pitch)
        croll = math.cos(roll)
        sroll = math.sin(roll)

        # Convert from Euler angles using the yaw-pitch-roll
        # Tait-Bryan convention.
        m00 = cyaw * cpitch
        m01 = syaw * sroll - cyaw * spitch * croll
        m02 = cyaw * spitch * sroll + syaw * croll
        m10 = spitch
        m11 = cpitch * croll
        m12 = -cpitch * sroll
        m20 = -syaw * cpitch
        m21 = syaw * spitch * croll + cyaw * sroll
        m22 = -syaw * spitch * sroll + cyaw * croll

        self._rotation_matrix[0, 0] = m00
        self._rotation_matrix[0, 1] = m01
        self._rotation_matrix[0, 2] = m02
        self._rotation_matrix[1, 0] = m10
        self._rotation_matrix[1, 1] = m11
        self._rotation_matrix[1, 2] = m12
        self._rotation_matrix[2, 0] = m20
        self._rotation_matrix[2, 1] = m21
        self._rotation_matrix[2, 2] = m22

        self._rodrigues_rotation_vector, jacobian = cv2.Rodrigues(
            self._rotation_matrix, self._rodrigues_rotation_vector)

    def _draw_object_axes(self):

        points_2D, jacobian = cv2.projectPoints(
            self._reference_axis_points_3D,
            self._rodrigues_rotation_vector,
            self._translation_vector, self._camera_matrix,
            self._distortion_coefficients)

        origin = (int(points_2D[0, 0, 0]), int(points_2D[0, 0, 1]))
        right = (int(points_2D[1, 0, 0]), int(points_2D[1, 0, 1]))
        up = (int(points_2D[2, 0, 0]), int(points_2D[2, 0, 1]))
        forward = (int(points_2D[3, 0, 0]), int(points_2D[3, 0, 1]))

        # Draw the X axis in red.
        cv2.arrowedLine(
            self._bgr_image, origin, right, (0, 0, 255), 2)

        # Draw the Y axis in green.
        cv2.arrowedLine(
            self._bgr_image, origin, up, (0, 255, 0), 2)

        # Draw the Z axis in blue.
        cv2.arrowedLine(
            self._bgr_image, origin, forward, (255, 0, 0), 2)

    def _make_and_draw_object_mask(self):

        # Project the object's vertices into the scene.
        vertices_2D, jacobian = cv2.projectPoints(
            self._reference_vertices_3D,
            self._rodrigues_rotation_vector,
            self._translation_vector, self._camera_matrix,
            self._distortion_coefficients)
        vertices_2D = vertices_2D.astype(numpy.int32)

        # Make a mask based on the projected vertices.
        self._mask.fill(0)
        for vertex_indices in \
                self._reference_vertex_indices_by_face:
            cv2.fillConvexPoly(
                self._mask, vertices_2D[vertex_indices], 255)

        # Draw the mask in semi-transparent yellow.
        cv2.subtract(
            self._bgr_image, 48, self._bgr_image, self._mask)


def main():
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    diagonal_fov_degrees = 70.0
    target_fps = 25.0

    demo = ImageTrackingDemo(
        capture, diagonal_fov_degrees, target_fps)
    demo.run()


if __name__ == '__main__':
    main()
