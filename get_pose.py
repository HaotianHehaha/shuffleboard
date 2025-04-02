import pyrealsense2 as rs
import numpy as np
import cv2
import json
import time
from pupil_apriltags import Detector

# Define tag size in meters
tag_size = 0.04  # 4 cm

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
    recording = False
    pose_data = np.zeros(3)
    rotation_matrix = np.zeros((3,3))

    for i in range(20):
        # Wait for a new frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame or i<10:
            continue
        
        # Get the timestamp of the current frame
        frame_timestamp = frames.get_timestamp()

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
            R = detection.pose_R  # Rotation matrix

            pose_data += t
            rotation_matrix += R

    # Cleanup
    pipeline.stop()
    cv2.destroyAllWindows()

    pose_camera  = pose_data/10
    rotation_matrix_camera = rotation_matrix/10

    pose_robot = np.dot(transform_matrix[:3,:3],pose_camera.reshape(3,1)).flatten() + transform_matrix[:3,3]
    rotation_matrix_robot = transform_matrix[:3,:3] @ rotation_matrix_camera 

    return pose_robot, rotation_matrix_robot
