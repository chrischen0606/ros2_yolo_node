import cv2
import cv2.aruco as aruco
import numpy as np
from std_msgs.msg import Float32MultiArray

class EdgeDetector:
    def __init__(self, ros_comunicator, image_processor):
        self.image_processor = image_processor
        self.ros_communicator = ros_comunicator
        self.color_bounds = {
            'door_edge':[np.array([10, 50, 50]), np.array([30, 255, 255])],
            'pole_edge':[np.array([0, 0, 200]), np.array([180, 15, 255])],
            # # 'pole_darkside':[np.array([96, 0, 133]), np.array([106, 82, 233])],
            'wall_edge':[np.array([86, 48, 103]), np.array([106, 148, 203])],
            # 'floor':[np.array([88, 96, 104]), np.array([108, 113, 119])]
        }
        self.rgb_map = {
            'wall': (103, 156, 174),
            'wall_dark': (104, 150, 163),
            'wall_light': (133, 200, 219),
            'white': (255, 255, 255),
            'white_shadow': (209, 226, 242),
            'floor_light': (108, 113, 119),
            'floor_dark': (88, 96, 104),
            'door': (180, 123, 80),
        }
        self.edge_colors = {
            'door_edge': (0, 255, 0),      # Green
            'pole_edge': (255, 0, 0),      # Blue
            'pole_darkside': (0, 0, 255),
            'wall_edge': (0, 255, 255),    # Yellow
            'floor': (255, 0, 255)    # Magenta
        }
        self.edge_msg = {}
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = self._create_detector_params()
        
    def _create_detector_params(self):
        params = cv2.aruco.DetectorParameters_create()
        params.adaptiveThreshWinSizeMin = 5
        params.adaptiveThreshWinSizeMax = 15
        params.adaptiveThreshWinSizeStep = 5
        params.cornerRefinementWinSize = 5
        params.perspectiveRemoveIgnoredMarginPerCell = 0.15
        return params
    # def edge_detection(image, color1, color2, tolerance1=(10, 40, 40), )
    def try_detect_edges(self):
        image = self.image_processor.get_rgb_cv_image()
    
        if image is None:
            print("No image available for ArUco detection.")
            return []
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # corners, ids, _ = aruco.detectMarkers(
        #     gray, self.aruco_dict, parameters=self.aruco_params
        # )
        # cv2.aruco.drawDetectedMarkers(image, corners, ids)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        for label, (lower, upper) in self.color_bounds.items():
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter out small fragments
            major_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]
            edge_msg = Float32MultiArray()
            
            for cnt in major_contours:
                # Polygon approximation to get corners
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                sorted_points = sorted(approx, key=lambda p: p[0][1], reverse=True)
                lower_two = sorted_points[:2]  # two points with largest y (lowest points)
                
                for point in lower_two:
                    pt = tuple(point[0])
                    cv2.circle(image, pt, 5, (255, 0, 0), -1)  # Blue dot
                    if label == 'door_edge':
                        cv2.putText(image, f"{pt}", (pt[0] + 5, pt[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                
                # Optionally, connect only the lower two corners with a line
                pt1 = tuple(lower_two[0][0])
                pt2 = tuple(lower_two[1][0])
                # print(pt1, pt2)
                cv2.line(image, pt1, pt2, self.edge_colors[label], 2)
                edge_msg.data.extend([
                    float(pt1[0]), float(pt1[1]),
                    float(pt2[0]), float(pt2[1])
                ])
            self.ros_communicator.publish_data(label, edge_msg)
            
        # def rgb_to_hsv_bgr(rgb):
        #     bgr = np.uint8([[rgb]])  # RGB to BGR because OpenCV uses BGR
        #     hsv = cv2.cvtColor(bgr, cv2.COLOR_RGB2HSV)
        #     return hsv[0][0]

        # # 1. Convert RGBs to HSV
        # pole_hsv = rgb_to_hsv_bgr(self.rgb_map['white'])
        # wall_hsv = rgb_to_hsv_bgr(self.rgb_map['wall'])

        # # 2. Define narrow HSV bounds
        # def narrow_bounds(hsv, h_tol=5, s_tol=40, v_tol=40):
        #     lower = np.array([
        #         max(hsv[0] - h_tol, 0),
        #         max(hsv[1] - s_tol, 0),
        #         max(hsv[2] - v_tol, 0)
        #     ])
        #     upper = np.array([
        #         min(hsv[0] + h_tol, 179),
        #         min(hsv[1] + s_tol, 255),
        #         min(hsv[2] + v_tol, 255)
        #     ])
        #     return lower, upper

        # pole_lower, pole_upper = narrow_bounds(pole_hsv)
        # wall_lower, wall_upper = narrow_bounds(wall_hsv)

        # # 3. Create HSV image
        # hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # # 4. Create binary masks
        # pole_mask = cv2.inRange(hsv_img, pole_lower, pole_upper)
        # wall_mask = cv2.inRange(hsv_img, wall_lower, wall_upper)

        # # 5. Dilate masks slightly to make them touch
        # kernel = np.ones((50, 50), np.uint8)
        # pole_mask = cv2.dilate(pole_mask, kernel, iterations=1)  # More iterations (was 1)
        # wall_mask = cv2.dilate(wall_mask, kernel, iterations=1)

        # # 6. Find overlapping borders
        # overlap = cv2.bitwise_and(pole_mask, wall_mask)

        # # 7. Find contours of overlap and draw them
        # contours, _ = cv2.findContours(overlap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # major_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]
        # cv2.drawContours(image, major_contours, -1, (0, 255, 255), 2)
        
        # Publish the edge image
        ros_image = self.image_processor.get_rgb_ros_image(image)
        self.ros_communicator.publish_data("edge_image", ros_image)
