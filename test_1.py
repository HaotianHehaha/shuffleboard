# test for mujoco

import mujoco
import mujoco.viewer
import numpy as np
import os

# 加载 URDF 文件（假设已经转换为 MJCF 格式）
urdf_path = "franka/franka.urdf"  # 你的URDF文件路径
model = mujoco.MjModel.from_xml_path(urdf_path)
data = mujoco.MjData(model)

# Franka 机械臂的 7 个关节索引（假设顺序为 Panda 关节）
franka_joint_names = [
    "panda_joint1", "panda_joint2", "panda_joint3",
    "panda_joint4", "panda_joint5", "panda_joint6",
    "panda_joint7"
]

# 获取关节索引
joint_ids = [model.joint(name).qposadr[0] for name in franka_joint_names]

# 设定初始关节角度（单位：弧度）
initial_joint_angles = np.array([0, -0.5, 0, -1.5, 0, 1.0, 0])

# 应用到 MuJoCo 数据结构
for i, joint_id in enumerate(joint_ids):
    data.qpos[joint_id] = initial_joint_angles[i]

# 计算初始状态
mujoco.mj_forward(model, data)

# 启动 MuJoCo 可视化界面
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)  # 物理仿真步进
        viewer.sync()


