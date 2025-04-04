import pyrealsense2 as rs
import numpy as np
import cv2
import json
import time
from pupil_apriltags import Detector

# Define tag size in meters
tag_size = 0.036  # 4 cm

# Initialize Intel RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Get camera intrinsics dynamically
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx, fy = intrinsics.fx, intrinsics.fy
cx, cy = intrinsics.ppx, intrinsics.ppy
camera_params = [fx, fy, cx, cy]

print(f"Camera Intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

# Initialize AprilTag detector
detector = Detector(families="tag36h11")

# Pose recording variables
recording = False
pose_data = []

while True:
    # Wait for a new frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
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
        # Draw tag border
        for idx in range(len(detection.corners)):
            pt1 = tuple(map(int, detection.corners[idx]))
            pt2 = tuple(map(int, detection.corners[(idx + 1) % 4]))
            cv2.line(color_image, pt1, pt2, (0, 255, 0), 2)

        # Draw center
        center = tuple(map(int, detection.center))
        cv2.circle(color_image, center, 5, (0, 0, 255), -1)

        # Display ID
        cv2.putText(color_image, f"ID: {detection.tag_id}",
                    (center[0] - 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Extract pose
        t = detection.pose_t.flatten()  # Translation vector
        R = detection.pose_R  # Rotation matrix

        # Display translation in GUI
        pose_text = f"XYZ: ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) m"
        cv2.putText(color_image, pose_text, (center[0] - 50, center[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Draw coordinate axes (X = red, Y = green, Z = blue)
        axis_length = 0.2  # Reduced axis length for better visualization
        axes = np.array([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])  # X, Y, Z
        axes_img_pts = []
        for i in range(3):
            axis_end = t + R @ axes[i]  # Transform axis endpoints
            x, y = int(fx * axis_end[0] / axis_end[2] + cx), int(fy * axis_end[1] / axis_end[2] + cy)
            axes_img_pts.append((x, y))

        cv2.line(color_image, center, axes_img_pts[0], (0, 0, 255), 2)  # X - Red
        cv2.line(color_image, center, axes_img_pts[1], (0, 255, 0), 2)  # Y - Green
        cv2.line(color_image, center, axes_img_pts[2], (255, 0, 0), 2)  # Z - Blue (pointing outward)

        # Print pose to console
        print(f"Tag ID: {detection.tag_id}")
        print(f"Translation (m): x={t[0]:.3f}, y={t[1]:.3f}, z={t[2]:.3f}")
        print(f"Rotation matrix:\n{R}\n")

        # Record pose if recording is active
        if recording:
            timestamp = frame_timestamp
            pose_data.append({
                "timestamp": timestamp,
                "tag_id": detection.tag_id,
                "translation": {"x": t[0], "y": t[1], "z": t[2]},
                "rotation_matrix": R.tolist()
            })

    # Show image
    cv2.imshow('AprilTag Pose Estimation', color_image)

    # Key press handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break  # Quit
    elif key == ord('r'):
        recording = True
        pose_data = []
        print("🔴 Recording started...")
    elif key == ord('s'):
        recording = False
        # Save to JSON
        filename = f"raw_data/apriltag_poses_{int(time.time())}.json"
        with open(filename, "w") as f:
            json.dump(pose_data, f, indent=4)
        print(f"🟢 Recording stopped. Data saved to {filename}")

# Cleanup
pipeline.stop()
cv2.destroyAllWindows()
