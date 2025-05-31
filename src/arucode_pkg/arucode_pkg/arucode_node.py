import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import cv2.aruco as aruco
import numpy as np

marker_length = 0.36  # meter
half_size = marker_length / 2
objp = np.array([
    [-half_size,  half_size, 0],  # top-left
    [ half_size,  half_size, 0],  # top-right
    [ half_size, -half_size, 0],  # bottom-right
    [-half_size, -half_size, 0]   # bottom-left
], dtype=np.float32)
class ArucoDetector(Node):
    def __init__(self):
        super().__init__("aruco_detector")

        # 訂閱相機影像
        self.image_sub = self.create_subscription(
            CompressedImage, "/camera/image/compressed", self.image_callback, 10
        )

        # 載入 ArUco 字典（使用 4x4_50 作為範例）
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = aruco.DetectorParameters_create()

    def image_callback(self, msg):
        try:
            # 轉換 ROS 影像訊息為 OpenCV 格式
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        # 偵測 ArUco 標記
        corners, ids, _ = aruco.detectMarkers(
            cv_image, self.aruco_dict, parameters=self.aruco_params
        )

        if ids is not None:
            detected_ids = ids.flatten().tolist()
            self.get_logger().info(f"Detected ArUco IDs: {detected_ids}")  # 列印 ID
            self.get_logger().info(f"Detected ArUco IDs: {corners[0]}")     

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
