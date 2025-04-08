import pinocchio as pin
import numpy as np
from scipy.optimize import minimize
import scipy
import subprocess
import re
import pdb

urdf_path = "franka/franka.urdf"  # 替换为你的 URDF 文件路径
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()

def compute_joint_v_from_ee(q,v_ee):
    # 计算雅可比矩阵
    frame_id = model.getFrameId("panda_ee")  # 末端执行器的 Frame ID
    pin.forwardKinematics(model, data, q)  # 计算正运动学
    J = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL)

    # 计算伪逆（Moore-Penrose 伪逆）
    J_pinv = scipy.linalg.pinv(J)  # 或者使用 scipy.linalg.pinv2() 以提高数值稳定性

    # 计算关节速度
    dq = J_pinv[:,:3] @ v_ee
    dq = dq.tolist() 
    return dq


def hitting_pose(position, rotation, bound_lower, bound_upper):

    # 定义末端执行器的帧 ID（通常是 "panda_link8" 或 "panda_hand"）
    end_effector_frame_id = model.getFrameId("panda_ee")  # 根据 URDF 中的帧名称修改
    

    # 定义期望的末端位姿（位置和姿态）
    target_position = np.array(position)  # 目标位置 (x, y, z)
    target_rotation = np.array(rotation) # 目标姿态（旋转矩阵），这里设置为单位矩阵（无旋转）

    # 定义逆运动学的目标函数
    def objective(q):
        # 更新机器人姿态
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacement(model, data, end_effector_frame_id)

        
        # 获取当前末端执行器的位姿
        current_placement = data.oMf[end_effector_frame_id]
        current_position = current_placement.translation-np.array([0.0, 0.0, 0.005])
        current_rotation = current_placement.rotation
        
        # 计算位置误差
        position_error = np.linalg.norm(current_position - target_position)
        
        # 计算姿态误差（旋转矩阵的差）
        rotation_error = np.linalg.norm(current_rotation - target_rotation)
        
        # 总误差（位置误差 + 姿态误差）
        total_error = position_error + rotation_error
        return total_error
    
    # 运行 echo_robot_state 获取机械臂关节角
    robot_id = '192.168.1.101'
    result = subprocess.run(
        ['/media/haotian/new_volumn/code/libfranka/build/examples/echo_robot_state', robot_id],  # 替换为实际路径
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )
    # 解析输出，提取 q 字段
    output = result.stdout
    # 使用正则表达式提取 "q" 的值
    match = re.search(r'"q":\s*\[([^\]]+)\]', output)
    if match:
        q_str = match.group(1)  # 提取 "q" 的值部分
        q_0 = np.array([float(x) for x in q_str.split(",")]+[0,0])  # 转换为 NumPy 数组
    else:
        raise ValueError("q not found in echo_robot_state output")
    while True:
        q1 = q_0 + np.random.randn(len(pin.neutral(model)))/2
 
        # print("Initial joint angles (rad):", q1.tolist())

        bounds =  zip(bound_lower, bound_upper)
        result = minimize(objective, q1, method='SLSQP', bounds=bounds, tol = 1e-5)
        if result.fun < 1e-4:
            break
    optimized_q = result.x
    print(f"Optimized distance: {result.fun:.2e}")
    print("Optimized joint angles (rad):", optimized_q.tolist())
    return optimized_q.tolist()

