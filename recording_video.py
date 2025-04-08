import pyrealsense2 as rs
import cv2
import numpy as np

def record_video():
    # 配置 RealSense 管道
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # 开启管道
    pipeline.start(config)

    print("Press 'R' to start recording, 'S' to stop recording, and 'Q' to quit.")

    # 初始化变量
    recording = False
    out = None

    try:
        while True:
            # 获取帧
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # 转换为 NumPy 数组
            color_image = np.asanyarray(color_frame.get_data())

            # 显示实时视频
            cv2.imshow('RealSense', color_image)

            # 检测键盘输入
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):  # 按下 'R' 键开始录制
                if not recording:
                    print("Recording started...")
                    recording = True
                    # 初始化视频写入器
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编码器
                    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1280, 720))

            elif key == ord('s'):  # 按下 'S' 键停止录制
                if recording:
                    print("Recording stopped.")
                    recording = False
                    if out:
                        out.release()
                        out = None

            elif key == ord('q'):  # 按下 'Q' 键退出
                print("Exiting...")
                break

            # 如果正在录制，将帧写入视频文件
            if recording and out:
                out.write(color_image)

    finally:
        # 释放资源
        if out:
            out.release()
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    record_video()