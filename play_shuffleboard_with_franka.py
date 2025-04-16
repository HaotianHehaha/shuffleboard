from hitting_planning import WarpFrankaEnv
from sysID import sysID
from tqdm import tqdm
import argparse
import numpy as np
from get_pose import get_position
import pdb
def main(args):
    
    transition_matrix = np.array([[-0.04779968605306184, 0.9963935576022187, -0.07010754868074545, 0.5694730814801501],
     [0.9988203701903047, 0.0470794467417468, -0.011890911966449851, -0.3497530708676511], 
     [-0.008547403473076184, -0.07059322958531787, -0.9974685648332899, 0.8198562497362987], 
     [0.0, 0.0, 0.0, 1.0]])
    

    target_position, _ = [ 0.54362988, -0.6218895 ,  0.11144529 ],[]#get_position(transition_matrix)
    # position1 : [ 0.54362988, -0.6218895 ,  0.11144529]
    # position2 : [ 0.54552426 ,-0.70059678,  0.1170872 ]
    # position3 : [ 0.5410096 , -0.77296564,  0.11270821]
    print(f'target_position: {target_position}')
    # pdb.set_trace()
    
    initial_position, rotation_matrix = get_position(transition_matrix)
    print(f'initial_position: {initial_position}')
    print(f'rotation_matrix: {rotation_matrix}')

    target_position[2] ,initial_position[2] = 0.085, 0.085


    
    example = sysID( filepath=args.filepath,verbose=args.verbose)

    # replay and optimize
    for i in range(args.train_iters_id):
        flag = example.step()
        if i == int(args.train_iters_id*1/4):
            # learning rate decay
            example.optimizer.lr /= 2
        elif i == int(args.train_iters_id*1/2):
            # learning rate decay
            example.optimizer.lr /= 2
        if flag:
            break

    mu = example.best_mu # 0.003 # 梯度爆炸
    loss = example.best_loss
    print(f'mu: {mu} loss:{loss} flag:{flag}')
    # mu = 0.01
    # pdb.set_trace()

    # ee_speed_init = 0.46
    ee_speed = 0.4
    lr = 0.1
    env = WarpFrankaEnv(stage_path_1=args.stage_path_1,stage_path_2=args.stage_path_2,integrator='featherstone',num_frames = args.num_frames, mu = mu, initial_position = initial_position,target_position = target_position)
    for _ in tqdm(range(args.train_iters_planning)):
        error,hitting_pose = env.step(ee_speed=ee_speed)  
        print(f'ee_speed:{ee_speed} error:{error}')
        if abs(error)<0.01:
            break
        ee_speed -= lr*error
        
    # error,hitting_pose = env.step(ee_speed=ee_speed)

    if env.renderer_1:
        env.renderer_1.save()
    if env.renderer_2:
        env.renderer_2.save()
    
    ee_speed_1 = env.evaluate_speed(ee_speed=env.best_ee_speed)
    print(f'real_ee_speed:{ee_speed_1} error:{error}') # 不清楚为什么会有gap？

    # 人工检查优化结果，从键盘读入数值赋给ee_speed_1，如果没有默认原来数值

    # input_speed = input('ee_speed_1:')
    # if input_speed != '':
    #     ee_speed_1 = float(input_speed)
    
    distance = np.linalg.norm(np.array(target_position)-np.array(initial_position))
    if distance > 0.68 and distance < 0.8:
        ee_speed_1 = np.clip(ee_speed_1, 0.34, 0.352)
    elif distance > 0.60:
        ee_speed_1 = np.clip(ee_speed_1, 0.328, 0.333)
    elif distance > 0.50:
        ee_speed_1 = np.clip(ee_speed_1, 0.305, 0.315)
    print(f'ee_speed_final: {ee_speed_1}')



    orientation = (np.array(target_position)-np.array(initial_position))/np.linalg.norm(np.array(target_position)-np.array(initial_position))
    ee_speed = ((ee_speed_1*2.5-0.365)*orientation).tolist()[:2]


    return ee_speed, hitting_pose

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--stage_path_1",
        type=lambda x: None if x == "None" else str(x),
        default=None,
        help="Path to the output USD file (before hitting).",
    )
    parser.add_argument(
        "--stage_path_2",
        type=lambda x: None if x == "None" else str(x),
        default=None,
        help="Path to the output USD file (after hitting).",
    )
    parser.add_argument(
        "--filepath",
        type=lambda x: None if x == "None" else str(x),
        default="raw_data/apriltag_poses_1741847062.json",
        help="Path to raw data file.",
    )
    parser.add_argument("--train_iters_id", type=int, default=40, help="Total number of training iterations for sysID")
    parser.add_argument("--train_iters_planning", type=int, default=5, help="Total number of training iterations for motion planning")
    parser.add_argument("--num_frames", type=int, default=1800, help="Number of frames to simulate")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    args = parser.parse_known_args()[0]

    ee_speed, hitting_pose = main(args)
    # if ee_speed<0:
    #     ee_speed = 0.0

    print(f'Max Speed: {",".join(map(str, ee_speed))}')
    print(f'Initial Joint Angles: {",".join(map(str, hitting_pose))}')