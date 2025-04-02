import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os
from pupil_apriltags import Detector
import subprocess

# Initialize the robot


# Define tag size in meters
tag_size = 0.06  # 64 mm

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
detector = Detector(families="tag25h9") #tag25h9

# Create a window
cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

# Initialize variables
data = []
intrinsics = None
image_counter = 0  # Counter for image filenames

def save_to_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
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
            # print(f"Tag ID: {detection.tag_id}")
            # print(f"Translation (m): x={t[0]:.3f}, y={t[1]:.3f}, z={t[2]:.3f}")
            # print(f"Rotation matrix:\n{R}\n")

            # Create homogeneous transformation matrix
            # 相机坐标到AprilTag坐标系的变换
            homogeneous_matrix = np.eye(4)
            homogeneous_matrix[:3, :3] = R
            homogeneous_matrix[:3, 3] = t

        # Show images
        cv2.imshow('RealSense', color_image)
        key = cv2.waitKey(1)

        if key == ord('r'):
        # 调用 echo_robot_state 获取机械臂的末端位姿
            robot_id = '192.168.1.101'
            try:
                result = subprocess.run(
                    ['/media/haotian/new_volumn/code/libfranka/build/examples/echo_robot_state', robot_id],  # 替换为实际路径
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
                # 解析输出，提取 O_T_EE 字段
                output = result.stdout
                for line in output.splitlines():
                    if "O_T_EE" in line:
                        matrix_str = line.split(":")[1].strip()  # 提取 O_T_EE 的值
                        matrix = eval(matrix_str)[0]  # 将字符串转换为 Python 列表
                        matrix = np.array(matrix).reshape((4, 4)).T
                        break
                else:
                    raise ValueError("O_T_EE not found in echo_robot_state output")

                print("pose_matrix: ", matrix)
                # Save the image to a file
                image_filename = f'{image_counter}.png'
                cv2.imwrite(image_filename, color_image)
                # Save the image filename, random matrix, and homogeneous transformation matrix as a pair
                data.append({
                    'image': image_filename,
                    'matrix': matrix.tolist(),
                    'homogeneous_matrix': homogeneous_matrix.tolist()
                })
                print(f"Image, matrix, and homogeneous matrix recorded: {image_filename}")
                image_counter += 1  # Increment the counter

            except subprocess.CalledProcessError as e:
                print(f"Error executing echo_robot_state: {e.stderr}")
            except Exception as e:
                print(f"Error processing echo_robot_state output: {e}")

        elif key == ord('q'):
            # Get intrinsics
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            intrinsics_data = {
                'width': intrinsics.width,
                'height': intrinsics.height,
                'ppx': intrinsics.ppx,
                'ppy': intrinsics.ppy,
                'fx': intrinsics.fx,
                'fy': intrinsics.fy,
                # 'model': intrinsics.model,
                # 'coeffs': intrinsics.coeffs
            }
            # Save data to JSON file
            save_to_json('data.json', {'data': data, 'intrinsics': intrinsics_data})
            print("Intrinsics saved and exiting.")
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()