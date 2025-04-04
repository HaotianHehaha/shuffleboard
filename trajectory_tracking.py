import json
import numpy as np
import pdb
from scipy.spatial.transform import Rotation 

def load_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def calculate_velocity(data):
    velocities = []
    for i in range(1, len(data)-1):
        t1, t2 = data[i-1]['timestamp'], data[i+1]['timestamp']
        dt = (t2 - t1)/1000
        
        pos1 = np.array([data[i-1]['translation']['x'], data[i-1]['translation']['y'], data[i-1]['translation']['z']])
        pos2 = np.array([data[i+1]['translation']['x'], data[i+1]['translation']['y'], data[i+1]['translation']['z']])
        
        velocity = (pos2 - pos1) / dt
        velocities.append({
            'timestamp': t2,
            'velocity': velocity.tolist()
        })
    return velocities

def calculate_angular_velocity(data):
    angular_velocities = []
    for i in range(1, len(data)-1):
        t1, t2 = data[i-1]['timestamp'], data[i+1]['timestamp']
        dt = (t2 - t1)/1000
        
        R1 = np.array(data[i-1]['rotation_matrix'])
        R2 = np.array(data[i+1]['rotation_matrix'])
        
        R = np.dot(R2, R1.T)
        angle_axis = rotation_matrix_to_angle_axis(R)
        
        angular_velocity = angle_axis / dt
        angular_velocities.append({
            'timestamp': t2,
            'angular_velocity': angular_velocity.tolist()
        })
    return angular_velocities

def rotation_matrix_to_angle_axis(R):
    angle = np.arccos((np.trace(R) - 1) / 2)
    if angle == 0:
        return np.zeros(3)
    axis = (1 / (2 * np.sin(angle))) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return angle * axis

def calculate_normal_vector(velocities):
    velocity_vectors = np.array([v['velocity'] for v in velocities])
    _, _, vh = np.linalg.svd(velocity_vectors, full_matrices=False)
    normal_vector = vh[-1]
    return normal_vector

def calculate_perpendicular_unit_vectors(normal_vector):
    # Find a vector that is not parallel to the normal vector
    if np.allclose(normal_vector, [1, 0, 0]):
        other_vector = np.array([0, 1, 0])
    else:
        other_vector = np.array([1, 0, 0])
    
    # Calculate the first perpendicular unit vector
    x_axis = np.cross(normal_vector, other_vector)
    x_axis /= np.linalg.norm(x_axis)
    
    # Calculate the second perpendicular unit vector
    y_axis = np.cross(normal_vector, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    
    return x_axis, y_axis

def transform_to_new_coordinate_system(data, origin, rotation_matrix):
    transformed_data = []
    rotation_matrix = rotation_matrix.T
    
    for frame in data:
        translation = np.array([frame['translation']['x'], frame['translation']['y'], frame['translation']['z']])
        rotation = np.array(frame['rotation_matrix'])
        
        # Transform the translation
        transformed_translation = np.dot(rotation_matrix, translation - origin)
        
        # Transform the rotation matrix
        transformed_rotation = np.dot(rotation_matrix, rotation)
        
        transformed_frame = {
            'timestamp': frame['timestamp'],
            'tag_id': frame['tag_id'],
            'translation': {
                'x': transformed_translation[0],
                'y': transformed_translation[1],
                'z': transformed_translation[2]
            },
            'rotation_matrix': transformed_rotation.tolist()
        }
        
        transformed_data.append(transformed_frame)
    
    return transformed_data

def fix_rotation_matrix(rotation_matrix):
    # 当前旋转矩阵的 Z 轴
    current_z_axis = rotation_matrix[:, 2]
    
    # 目标 Z 轴
    target_z_axis = np.array([0, 0, 1])
    
    # 计算旋转轴
    rotation_axis = np.cross(current_z_axis, target_z_axis)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    # 计算旋转角度
    cos_theta = np.dot(current_z_axis, target_z_axis)
    theta = np.arccos(cos_theta)
    
    # 计算修正旋转矩阵
    correction_rotation = Rotation.from_rotvec(rotation_axis * theta).as_matrix()
    
    # 修正原始旋转矩阵
    fixed_rotation_matrix = correction_rotation @ rotation_matrix
    
    return fixed_rotation_matrix

def tracking(filepath = 'apriltag_poses_1741847062.json'):
    data = load_data(filepath)
    
    # Calculate the average origin and rotation matrix
    origin = np.zeros(3)
    quaternions = []
    for i in range(10):       
        origin += np.array([data[i]['translation']['x'], data[i]['translation']['y'], data[i]['translation']['z']])
        quaternion = Rotation.from_matrix(data[i]['rotation_matrix']).as_quat()
        quaternions.append(quaternion)
    origin /= 10
    avg_quaternion = np.mean(quaternions, axis=0)
    R = Rotation.from_quat(avg_quaternion).as_matrix() 
    
    transformed_data = transform_to_new_coordinate_system(data, origin, R)
    
    transformed_velocities = calculate_velocity(transformed_data)
    # transformed_angular_velocities = calculate_angular_velocity(transformed_data)
    v = []
    for i in range(len(transformed_velocities) ):
        transformed_data[i]['velocity'] = transformed_velocities[i]['velocity']
        # transformed_data[i]['angular_velocity'] = transformed_angular_velocities[i]['angular_velocity']
        v.append(np.linalg.norm(np.array(transformed_velocities[i]['velocity'])))
    
    # get initial data
    # print(np.array([transformed_data[0]['translation']['x'], transformed_data[0]['translation']['y'], transformed_data[0]['translation']['z']]))

    index = np.argmax(v) -1
    initial_position = np.array([transformed_data[index]['translation']['x'], transformed_data[index]['translation']['y'], transformed_data[index]['translation']['z']])
    initial_orientation = np.array(transformed_data[index]['rotation_matrix'])
    initial_orientation = fix_rotation_matrix(initial_orientation)
    initial_quaternion = Rotation.from_matrix(initial_orientation).as_quat()
    initial_velocity = np.array(transformed_velocities[index]['velocity'])
    initial_velocity[-1] = 0.0

    # initial_angular_velocity = np.array(transformed_angular_velocities[index]['angular_velocity'])
    initial_time = transformed_data[index]['timestamp']

    # get final data
    # noise = np.array(v[-10:]).mean()
    # for idx in range(index, len(v)):
    #     if v[idx] < noise:
    #         final_idx = idx +1
    #         break

    final_position = np.zeros(3)
    # final_orientation = np.zeros((3, 3))
    for i in range(1,11):
        final_position += np.array([transformed_data[-i]['translation']['x'], transformed_data[-i]['translation']['y'], transformed_data[-i]['translation']['z']])
        # final_orientation += np.array(transformed_data[-i]['rotation_matrix'])
    final_position /= 10
    # final_orientation /= 10
    # final_quaternion = Rotation.from_matrix(final_orientation).as_quat()
    # final_velocity = np.array(transformed_velocities[final_idx]['velocity'])
    # final_angular_velocity = np.array(transformed_angular_velocities[final_idx]['angular_velocity'])
    # final_time = transformed_data[final_idx]['timestamp']

    # print(f"Initial Position: {initial_position}")
    # print(f"Initial Velocity: {initial_velocity}")
    # print(f"Initial Angular Velocity: {initial_angular_velocity}")
    # print(f"Initial Time: {initial_time}")
    # print() 
    # print(f"Final Position: {final_position}")
    
    # print(f"Final Velocity: {final_velocity}")
    # print(f"Final Angular Velocity: {final_angular_velocity}")
    # print(f"Final Time: {final_time}")

    real_setting = {}
    real_setting['initial_position'] = initial_position.tolist()
    real_setting['initial_velocity'] = initial_velocity.tolist()
    # real_setting['initial_angular_velocity'] = initial_angular_velocity.tolist()
    real_setting['initial_quaternion'] = initial_quaternion.tolist()
    real_setting['initial_time'] = initial_time
    real_setting['final_position'] = final_position.tolist()
    # real_setting['final_quaternion'] = final_quaternion.tolist()
    # real_setting['final_velocity'] = final_velocity.tolist()
    # real_setting['final_angular_velocity'] = final_angular_velocity.tolist()
    # real_setting['final_time'] = final_time

    return real_setting



    # import matplotlib.pyplot as plt
    # plt.plot(v)
    # plt.show()


    # with open('transformed_apriltag_poses.json', 'w') as file:
    #     json.dump(transformed_data, file, indent=4)

    # for v in velocities:
    #     print(np.dot(v['velocity'],normal_vector))
    
    # for v,av in zip(transformed_velocities, transformed_angular_velocities):
    #     if np.linalg.norm(v['velocity']) > 0.05:
    #         print(f"Timestamp: {v['timestamp']}")
    #         print(f"Velocity: {v['velocity']}")
    #         print(f"Angular Velocity: {av['angular_velocity']}")
    #         print()

if __name__ == "__main__":
    tracking()