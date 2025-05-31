from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import torch


class CameraDepth():
    def __init__(self, ros_communication, image_processor):
        self.ros_communicator = ros_communication
        self.image_processor = image_processor

        depth_cv_image = self.image_processor.get_depth_cv_image()
        if depth_cv_image is None or not isinstance(depth_cv_image, np.ndarray):
            print("Depth image is invalid.")
        # 訂閱影像 Topic
        image = self.image_processor.get_rgb_cv_image()
        if image is None:
            print("No image available for ArUco detection.")

        
        # 相機畫面中央高度上切成 n 個等距水平點。
        self.x_num_splits = 20
        # self.publish_x_multi_depths(image)

    def get_depth_at(self, x, y):
        """
        取得指定像素的深度值，轉換為米 (m)
        若深度出問題，回傳 -1
        """
        # **優先使用無壓縮的深度圖**
        depth_image = self.latest_depth_image_raw
        
        if depth_image is None:
            return -1.0

        # 如果深度影像為三通道，那只取第一個數值
        if len(depth_image.shape) == 3:
            depth_image = depth_image[:, :, 0]

        try:
            depth_value = depth_image[y, x]
            if depth_value < 0.0001 or depth_value == 0.0:  # 無效深度
                return -1.0
            return depth_value / 1000.0  # 16-bit 深度圖通常單位為 mm，轉換為 m
        except IndexError:
            return -1.0
        
    def publish_x_multi_depths(self, image):
        """
        取得畫面 n 個等分點的深度並發布
        """
        height, width = image.shape[:2]
        cy_center = height // 2  # 固定 Y 座標在畫面中心
        segment_length = width // self.x_num_splits

        # 計算 10 個等分點的 X 座標
        points = [(i * segment_length, cy_center) for i in range(self.x_num_splits)]

        # 取得每個等分點的深度值
        depth_values = [self.get_depth_at(x, cy_center) for x, _ in points]

        # 以 Float32MultiArray 發布
        depth_msg = Float32MultiArray()
        depth_msg.data = depth_values
        self.ros_communicator.publish_data('x_multi_depth_pub', depth_msg)
