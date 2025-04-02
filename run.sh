#!/bin/bash

# 运行 Python 脚本并捕获输出
output=$(python play_shuffleboard_with_franka.py --stage_path 'franka_shuffleboard.usda' --filepath "raw_data/apriltag_poses_1741846916.json" --verbose)

# 从 Python 脚本的输出中提取末端执行器速度和关节角
# 假设 Python 脚本输出格式如下：
# Max Speed: 0.5
# Initial Joint Angles: 0.308,0.265,-0.271,-2.553,0.212,2.802,0.633

# 提取最大速度
max_speed=$(echo "$output" | grep "Max Speed" | awk '{print $3}')

# 提取初始关节角
initial_joint_angles=$(echo "$output" | grep "Initial Joint Angles" | awk -F': ' '{print $2}')

# 检查是否成功提取到变量
if [[ -z "$max_speed" || -z "$initial_joint_angles" ]]; then
  echo "Error: Failed to extract max speed or initial joint angles from Python script output."
  exit 1
fi

# 打印提取的值（可选，用于调试）
echo "Extracted Max Speed: $max_speed"
echo "Extracted Initial Joint Angles: $initial_joint_angles"

# cd ..
# cd /media/haotian/new_volumn/code/libfranka/build
# # 运行编译好的 C++ 程序，并将提取的值作为参数传递
# ./examples/generate_cartesian_velocity_motion 192.168.1.101 "$max_speed" "$initial_joint_angles"