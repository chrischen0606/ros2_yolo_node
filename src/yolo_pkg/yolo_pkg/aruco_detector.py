import contextlib
import io
import cv2
import cv2.aruco as aruco
import numpy as np
import yaml
import os
from scipy.spatial.transform import Rotation as scipy_R
from collections import deque
from sensor_msgs.msg import CompressedImage, Image
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header, Int32MultiArray
from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped

half_size = 0.17
objp = np.array([
    [-half_size,  half_size, 0],  # top-left
    [ half_size,  half_size, 0],  # top-right
    [ half_size, -half_size, 0],  # bottom-right
    [-half_size, -half_size, 0]   # bottom-left
], dtype=np.float32)

def get_transform_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

class ArucoDetector:
    def __init__(self, ros_comunicator, image_processor, load_params):
        self.image_processor = image_processor
        self.load_params = load_params
        self.ros_communicator = ros_comunicator
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = self._create_detector_params()
        self.camera_matrix = np.array([
            [576.83946  , 0.0       , 319.59192 ],
            [0.         , 577.82786 , 238.89255 ],
            [0.         , 0.        , 1.        ]
        ])
        self.dist_coeffs = np.array([0.001750, -0.003776, -0.000528, -0.000228, 0.000000])
        self.marker_length =0.34 # meter
        half_size = self.marker_length / 2
        self.objp = np.array([
            [-half_size,  half_size, 0],  # top-left
            [ half_size,  half_size, 0],  # top-right
            [ half_size, -half_size, 0],  # bottom-right
            [-half_size, -half_size, 0]   # bottom-left
        ], dtype=np.float32)
        self.info = None
        self.map_metadata = self.load_map()
        self.map_msg = self.setup_occupany_grid()
        self.timer = self.ros_communicator.create_timer(1.0, self.publish_map)
        self.aruco_positions = self.load_aruco_positions()
        self.cam_pose = None
        self.T_camera_laser = np.array([
            [0, 0, 1, -0.4],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])
        R_align = np.array([
            [ 0,  0,  1],  # map x <- camera z
            [-1,  0,  0],  # map y <- -camera x
            [ 0, -1,  0]   # map z <- -camera y
        ])
        self.T_align = np.eye(4)
        self.T_align[0:3, 0:3] = np.linalg.inv(R_align)
        self.reset_internal_state()
        # self.ros_communicator.create_timer(0.1, self.publish_pose)
        self.aruco_msg = Int32MultiArray()
        
    def _create_detector_params(self):
        params = cv2.aruco.DetectorParameters_create()
        params.adaptiveThreshWinSizeMin = 5
        params.adaptiveThreshWinSizeMax = 15
        params.adaptiveThreshWinSizeStep = 5
        params.cornerRefinementWinSize = 5
        params.perspectiveRemoveIgnoredMarginPerCell = 0.15
        return params

    def load_map(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        map_path = os.path.join(current_dir, '../map01')
        yaml_path = os.path.join(map_path, 'map01.yaml')
        with open(yaml_path, 'r') as f:
            map_metadata = yaml.safe_load(f)
        return map_metadata 

    def publish_map(self):
        self.ros_communicator.publish_data('occupancy_grid', self.map_msg)  
        
    def publish_pose(self):
        pose_msg = self.setup_pose_msg(self.last_pos, self.last_quat)
        self.ros_communicator.publish_data('car_pose', pose_msg)
        
    def publish_aruco_detection(self):
        ros_image = self.image_processor.get_rgb_ros_image(self.image)
        self.ros_communicator.publish_data("aruco_image", ros_image)
        
    def load_aruco_positions(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        aruco_path = os.path.join(current_dir, '../config/aruco_location.yaml')
        with open(aruco_path, 'r') as f:
            aruco_positions = yaml.safe_load(f)
        return aruco_positions
    
    def setup_occupany_grid(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image = cv2.imread(os.path.join(current_dir, f'../map01/{self.map_metadata["image"]}'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.flip(image, 0)
        map_msg = OccupancyGrid()
        map_msg.header = Header()
        map_msg.header.frame_id = 'map'
        map_msg.info.resolution = self.map_metadata['resolution']
        map_msg.info.width = image.shape[1]
        map_msg.info.height = image.shape[0]
        map_msg.info.origin.position.x = self.map_metadata['origin'][0]
        map_msg.info.origin.position.y = self.map_metadata['origin'][1]
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0
        data = []
        for row in image:
            for pixel in row:
                occ = 100 if pixel < self.map_metadata['occupied_thresh'] * 255 else 0
                data.append(occ)
        map_msg.data = data
        return map_msg
        
    def setup_pose_msg(self, pos, q):
        # PoseWithCovarianceStamped
        print(f"{pos}")
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.pose.position.x = pos[0]
        pose_msg.pose.pose.position.y = pos[1]
        pose_msg.pose.pose.position.z = 0.0

        pose_msg.pose.pose.orientation.x = q[0]
        pose_msg.pose.pose.orientation.y = q[1]
        pose_msg.pose.pose.orientation.z = q[2]
        pose_msg.pose.pose.orientation.w = q[3]
        print("pose estimatation!!")
        return pose_msg
    
    def cal_marker_to_map_matrix(self, marker_2d_info):
        # Build 4x4 marker_to_map matrix from 2D pose (z=0, flat on ground)
        cos_theta = np.cos(marker_2d_info['theta'])
        sin_theta = np.sin(marker_2d_info['theta'])
        marker_to_map = np.eye(4)
        marker_to_map[0, 0] = cos_theta
        marker_to_map[0, 1] = -sin_theta
        marker_to_map[1, 0] = sin_theta
        marker_to_map[1, 1] = cos_theta
        marker_to_map[0, 3] = marker_2d_info['x']
        marker_to_map[1, 3] = marker_2d_info['y']
        marker_to_map[2, 3] = 0.0
        return marker_to_map
    
    def polygon_area(self, pts):
        pts = np.array(pts[0])
        return 0.5 * abs(np.dot(pts[:,0], np.roll(pts[:,1], 1)) - np.dot(pts[:,1], np.roll(pts[:,0], 1)))

    def reset_internal_state(self):
        self.last_pos = np.zeros((3,))
        self.last_quat = np.zeros((4,))
        self.odom_pose = [0.0, 0.0, 0.0]
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.pose.position.x = self.last_pos[0]
        pose_msg.pose.pose.position.y = self.last_pos[1]
        pose_msg.pose.pose.position.z = self.last_pos[2]
        pose_msg.pose.pose.orientation.x = self.last_quat[0]
        pose_msg.pose.pose.orientation.y = self.last_quat[1]
        pose_msg.pose.pose.orientation.z = self.last_quat[2]
        pose_msg.pose.pose.orientation.w = self.last_quat[3]
        self.ros_communicator.publish_data('car_pose', pose_msg)

    def try_detect_aruco_marker(self):
        image = self.image_processor.get_rgb_cv_image()
        self.image = image
        if image is None:
            print("No image available for ArUco detection.")
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )
        for c in corners:
            cv2.cornerSubPix(
                gray, c,
                winSize=(5, 5), zeroZone=(-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
            )

        if ids is None or len(ids) == 0:
            print("[warning] No ArUco markers detected.")
            return
        
        
        self.aruco_msg.data = [int(i) for i in ids]
        self.ros_communicator.publish_data('aruco_marker', self.aruco_msg)
        print(self.aruco_msg)

        pose_list = []
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id not in self.aruco_positions or self.polygon_area(corners[i]) < 1500:
                continue

            image_points = corners[i]
            if marker_id not in [3, 4, 5]:
                reorder = [2, 3, 0, 1]
                image_points = image_points[:, reorder, :]

            success, rvec, tvec, _ = cv2.solvePnPRansac(
                self.objp, image_points,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=3.0, confidence=0.99, iterationsCount=100
            )
            if not success:
                continue

            T_camera_marker = get_transform_matrix(rvec, tvec)
            T_marker_camera = np.linalg.inv(T_camera_marker)
            T_map_marker = self.cal_marker_to_map_matrix(marker_2d_info=self.aruco_positions[marker_id])

            T_map_camera = T_map_marker @ np.linalg.inv(self.T_align) @ T_marker_camera
            T_map_base = T_map_camera @ np.linalg.inv(self.T_camera_laser)

            pos = T_map_base[:3, 3]
            R_matrix = T_map_base[:3, :3]
            q = scipy_R.from_matrix(R_matrix).as_quat()

            cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)
            pose_list.append(np.concatenate([pos, q]))

        ros_image = self.image_processor.get_rgb_ros_image(image)
        self.ros_communicator.publish_data("aruco_image", ros_image)
        
        if len(pose_list) == 0:
            print("[warning] No valid marker poses after filtering.")
            return

        pos_arr = np.array([pose[:3] for pose in pose_list])
        q_arr = np.array([pose[3:] for pose in pose_list])
        
        if hasattr(self, "last_pos"):
            dists = np.linalg.norm(pos_arr - self.last_pos, axis=1)
            filtered_idx = np.where(dists < 5)[0]
            if len(filtered_idx) == 0:
                pose_msg = self.setup_pose_msg(self.last_pos, self.last_quat)
                self.ros_communicator.publish_data('car_pose', pose_msg)
                print("[warning] All marker poses filtered out due to large displacement.")
                return
            pos_arr = pos_arr[filtered_idx]
            q_arr = q_arr[filtered_idx]

        avg_pos = np.mean(pos_arr, axis=0)
        avg_quat = np.mean(q_arr, axis=0)
        avg_quat /= np.linalg.norm(avg_quat)    

        self.last_pos = avg_pos
        self.last_quat = avg_quat

        pose_msg = self.setup_pose_msg(self.last_pos, self.last_quat)
        self.ros_communicator.publish_data('car_pose', pose_msg)
        
        
