# def map_2D_points_onto_3D_plane(points_2D, image_size,
#                                 image_real_height):
#     w, h = image_size
#     # 计算图像的缩放比例
#     image_scale = image_real_height / h
#
#     points_3D = []
#     for point_2D in points_2D:
#         x, y = point_2D
#         # 计算出三维平面上的点的坐标。
#         # x 坐标等于图像缩放比例乘以（x 减去图像宽度的一半），
#         # y 坐标等于图像缩放比例乘以（y 减去图像高度的一半），
#         # z 坐标为 0。
#         point_3D = (image_scale * (x - 0.5 * w),
#                     image_scale * (y - 0.5 * h),
#                     0.0)
#         points_3D.append(point_3D)
#     return numpy.array(points_3D, numpy.float32)
