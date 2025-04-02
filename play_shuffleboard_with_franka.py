from hitting_planning import WarpFrankaEnv
from sysID import sysID
from tqdm import tqdm
import argparse
import numpy as np
from get_pose import get_position
import pdb
def main(args):
    
    example = sysID( filepath=args.filepath,verbose=args.verbose)

    # replay and optimize
    for _ in range(args.train_iters_id):
        flag = example.step()
        if flag:
            break

    mu = example.model.shape_materials.mu.numpy()[0] # 0.003 # 梯度爆炸
    print(f'mu: {mu} flag:{flag}')
    if not flag:
        raise Exception('优化不足')
    
    transition_matrix = np.array([[-0.99148776, -0.02689386,  0.12739209 , 0.70049409],
                        [-0.09142382,  0.84045097 ,-0.5341197 ,  0.35703603],
                        [-0.09270227, -0.54121982 ,-0.83575559 , 0.74178218],
                        [ 0.    ,      0.   ,       0.   ,       1.        ]])
    initial_position, rotation_matrix = get_position(transition_matrix)
    print(f'initial_position: {initial_position}')
    print(f'rotation_matrix: {rotation_matrix}')

    pdb.set_trace()

    ee_speed = 0.1
    lr = 0.1
    env = WarpFrankaEnv(stage_path=args.stage_path,integrator='featherstone',num_frames = args.num_frames, mu = mu, initial_position = [0.5,0.0,0.085],target_position = [1.0, 0.0,0.085])
    for _ in tqdm(range(args.train_iters_planning)):
        error,hitting_pose = env.step(ee_speed=ee_speed)  
        
        if abs(error)<0.01:
            break
        ee_speed -= lr*error
        print(ee_speed)
        
        if env.renderer:
            env.renderer.save()

    # ee_speed =  0.354888830780983
    # hitting_pose = [0.5304272374900205,0.24681288471690424,-0.4785124188929678,-2.553576946694635,0.30490096334617595,2.757475534339315,0.5658752832698344,0.0,0.0]

    return ee_speed, hitting_pose

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default=None,
        help="Path to the output USD file.",
    )
    parser.add_argument(
        "--filepath",
        type=lambda x: None if x == "None" else str(x),
        default="raw_data/apriltag_poses_1741847062.json",
        help="Path to raw data file.",
    )
    parser.add_argument("--train_iters_id", type=int, default=30, help="Total number of training iterations for sysID")
    parser.add_argument("--train_iters_planning", type=int, default=5, help="Total number of training iterations for motion planning")
    parser.add_argument("--num_frames", type=int, default=600, help="Number of frames to simulate")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    args = parser.parse_known_args()[0]

    ee_speed, hitting_pose = main(args)
    print(f'Max Speed: {ee_speed}')
    print(f'Initial Joint Angles: {",".join(map(str, hitting_pose))}')