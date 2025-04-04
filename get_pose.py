import pyrealsense2 as rs
import numpy as np
import cv2
import json
import time
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R

# Define tag size in meters
tag_size = 0.036  # 4 cm

# Initialize Intel RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

def get_position(transform_matrix):
    # Start streaming
    profile = pipeline.start(config)

    # Get camera intrinsics dynamically
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy
    camera_params = [fx, fy, cx, cy]

    # Initialize AprilTag detector
    detector = Detector(families="tag36h11")

    # Pose recording variables
    pose_data = np.zeros(3)
    quaternions = []  # 用于存储四元数

    for i in range(20):
        # Wait for a new frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame or i < 10:
            continue

        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Convert to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags with pose estimation
        detections = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)

        # Draw detection results
        for detection in detections:
            # Extract pose
            t = detection.pose_t.flatten()  # Translation vector
            R_matrix = detection.pose_R  # Rotation matrix

            pose_data += t

            # 将旋转矩阵转换为四元数并存储
            quaternion = R.from_matrix(R_matrix).as_quat()
            quaternions.append(quaternion)

    # Cleanup
    pipeline.stop()
    cv2.destroyAllWindows()

    # 平均平移向量
    pose_camera = pose_data / 10

    # 平均四元数
    avg_quaternion = np.mean(quaternions, axis=0)
    avg_rotation_matrix = R.from_quat(avg_quaternion).as_matrix()  # 将平均四元数转换回旋转矩阵

    # 转换到机器人坐标系
    pose_robot = np.dot(transform_matrix[:3, :3], pose_camera.reshape(3, 1)).flatten() + transform_matrix[:3, 3]
    rotation_matrix_robot = transform_matrix[:3, :3] @ avg_rotation_matrix

    return pose_robot, rotation_matrix_robot
