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
    

    target_position, _ = [ 0.49997284, -0.75593994 , 0.11016196],[]#get_position(transition_matrix)
    print(f'target_position: {target_position}')
    
    # pdb.set_trace()
    initial_position, rotation_matrix = get_position(transition_matrix)
    print(f'initial_position: {initial_position}')
    print(f'rotation_matrix: {rotation_matrix}')

    # real_length = 0.8
    # scaling_factor = real_length/(target_position[1]-initial_position[1])
    # target_position[1] = target_position[1]*scaling_factor
    # initial_position[1] = initial_position[1]*scaling_factor
    target_position[2] ,initial_position[2] = 0.085, 0.085
    # target_position[0] += 0.03
    # initial_position[0]  += 0.03

    
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
    # if not flag:
    #     raise Exception('优化不足')

    ee_speed = 0.1
    lr = 0.1
    env = WarpFrankaEnv(stage_path=args.stage_path,integrator='featherstone',num_frames = args.num_frames, mu = mu, initial_position = initial_position,target_position = target_position)
    for _ in tqdm(range(args.train_iters_planning)):
        error,hitting_pose = env.step(ee_speed=ee_speed)  
        
        if abs(error)<0.01:
            break
        ee_speed -= lr*error
        print(ee_speed)
        
        if env.renderer:
            env.renderer.save()

    ee_speed =  0.1
    orientation = (np.array(target_position)-np.array(initial_position))/np.linalg.norm(np.array(target_position)-np.array(initial_position))
    ee_speed = (ee_speed*orientation).tolist()[:2]

    hitting_pose = [1.5811952631998871,1.720540765272226,-1.5289289973142626,-2.2610569998874555,1.7129416250052567,1.5076810532416658,1.6751969048387967,0.0,0.0]

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
    parser.add_argument("--train_iters_id", type=int, default=40, help="Total number of training iterations for sysID")
    parser.add_argument("--train_iters_planning", type=int, default=5, help="Total number of training iterations for motion planning")
    parser.add_argument("--num_frames", type=int, default=600, help="Number of frames to simulate")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    args = parser.parse_known_args()[0]

    ee_speed, hitting_pose = main(args)
    # if ee_speed<0:
    #     ee_speed = 0.0

    print(f'Max Speed: {",".join(map(str, ee_speed))}')
    print(f'Initial Joint Angles: {",".join(map(str, hitting_pose))}')