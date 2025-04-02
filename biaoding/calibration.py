import json
import cv2
import numpy as np

def load_matrices(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    matrices = []
    homogeneous_matrices = []
    
    for entry in data['data']:
        matrices.append(np.array(entry['matrix']))
        homogeneous_matrices.append(np.array(entry['homogeneous_matrix']))
    
    return matrices, homogeneous_matrices

def perform_hand_eye_calibration(matrices, homogeneous_matrices):
    # Convert lists to numpy arrays
    R_gripper2base = [np.array(matrix[:3,:3]) for matrix in matrices]
    t_gripper2base = [np.array(matrix[:3, 3]) for matrix in matrices]
    
    R_base2gripper = [R.T for R in R_gripper2base]
    t_base2gripper = [-R.T @ t for R, t in zip(R_gripper2base, t_gripper2base)]

    
    R_target2cam = [np.array(matrix[:3,:3]) for matrix in homogeneous_matrices]
    t_target2cam = [np.array(matrix[:3, 3]) for matrix in homogeneous_matrices]
    
    # Perform hand-eye calibration
    R_cam2gbase, t_cam2base = cv2.calibrateHandEye(
        R_base2gripper, t_base2gripper, R_target2cam, t_target2cam
    )
    
    # Create homogeneous transformation matrix
    cam2base_transformation_matrix = np.eye(4)
    cam2base_transformation_matrix[:3, :3] = R_cam2gbase
    cam2base_transformation_matrix[:3, 3] = t_cam2base.flatten()  # Flatten the column vector
            
    return cam2base_transformation_matrix

# Example usage
file_path = './data.json'
matrices, homogeneous_matrices = load_matrices(file_path)

cam2base_transformation_matrix = perform_hand_eye_calibration(matrices, homogeneous_matrices)

print("Transformation from base to camera is:")
print(cam2base_transformation_matrix)




