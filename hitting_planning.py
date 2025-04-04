# 击打模拟  
from IK_initialization import hitting_pose, compute_joint_v_from_ee
import warp as wp
import numpy as np

import warp.sim
import warp.sim.render
import warp.tape

from copy import deepcopy

URDF_PATH = "franka/franka.urdf"

@wp.kernel
def loss_kernel(body_q: wp.array(dtype=wp.transform), target:  wp.vec3, loss: wp.array(dtype=float)):

    x = wp.transform_get_translation(body_q[-1])
    # distance to target
    delta = x- target
    loss[0] = wp.dot(delta , delta)

@wp.kernel
def velocity_loss_kernel(body_qd: wp.array(dtype=wp.spatial_vector), target_speed: wp.vec3, loss: wp.array(dtype=float)):

    x = wp.spatial_bottom(body_qd[10])
    loss[0] = wp.dot(x , target_speed)


class WarpFrankaEnv():
    def __init__(self,stage_path='franka_shuffleboard.usda',integrator='featherstone',num_frames = 60, mu = 0.05, 
                 initial_position = [0.5,0.0,0.080],target_position = [1.0, 0.0,0.080]):
        super(WarpFrankaEnv).__init__()

        self.num_frames = num_frames
        # control frequency
        fps = 60
        self.frame_dt = 1.0 / fps

        # sim frequency
        self.sim_substeps_1 = 50
        self.sim_dt_1 = self.frame_dt / self.sim_substeps_1
        self.sim_substeps_2 = 8
        self.sim_dt_2 = self.frame_dt / self.sim_substeps_2

        self.iter = 0
        self.render_time = 0.0
        self.ke = 1e4
        self.kd = 10.0
        self.kf = 100.0 
        self.mu = mu

        self.shuffboard_size = [0.15, 0.6 ]
        self.board_initial_position = [0.48, -0.3, 0.065]
        self.box_size = [0.02,0.02,0.02]
        self.initial_position = initial_position

        self.target = target_position
        self.loss = wp.zeros(1, dtype=float,requires_grad=True)     

        self.reset(URDF_PATH)

        norm = np.linalg.norm(np.array(self.target)-np.array(self.initial_position))
        self.orientation = (np.array(self.target)-np.array(self.initial_position))/norm
        rotation_matrix = np.zeros((3,3))
        rotation_matrix[:,0] = self.orientation
        rotation_matrix[:,2] = np.array([0.0, 0.0, -1.0])
        rotation_matrix[:,1] = np.cross( np.array([0.0, 0.0, -1.0]),self.orientation)
        self.joint_q_franka =  hitting_pose(np.array(self.initial_position)-0.05*self.orientation, rotation_matrix, self.model.joint_limit_lower.numpy(), self.model.joint_limit_upper.numpy())
        self.joint_q_franka[-2:] = [0.0,0.0]
        
        # set ground parameters
        self.model.shape_materials.ke.fill_(self.ke)
        self.model.shape_materials.kd.fill_(self.kd)
        self.model.shape_materials.kf.fill_(self.kf)
        self.model.shape_materials.restitution.fill_(0.0)
        self.model.shape_materials.mu.fill_(self.mu)
        self.model.rigid_contact_margin = 1e-3
        
        self.initial_state = self.model.state()
        self.state_sequence = []
        
        self.model.joint_q.assign(self.model.joint_q.numpy().tolist()[:7] + self.joint_q_franka)

        # set different gravity for franka
        gravity_normal = self.model.gravity.reshape(1, 3)
        gravity_articuation = np.array([[0., 0., 0.]], dtype=object)
        self.model.gravity = wp.array(np.concatenate([gravity_normal, gravity_articuation],axis=0),dtype=wp.vec3)

        if integrator == 'featherstone':
            self.integrator = warp.sim.FeatherstoneIntegrator(self.model)
        elif integrator == 'semi-implicit':
            self.integrator = warp.sim.SemiImplicitIntegrator()
        elif integrator == 'xpbd':
            self.integrator = warp.sim.XPBDIntegrator(iterations=2)
        else:
            raise ValueError(f"Unknown integrator: {integrator}")
        
        self.model_simple = self.simple_model()
        self.integrator_fast = warp.sim.FeatherstoneIntegrator(self.model_simple)

        if stage_path:
            self.renderer = warp.sim.render.SimRenderer(self.model_simple, stage_path, scaling=1.0)

        
        self.flag = 0 # 0 represent before hitting time  1 represent after hitting time
    
    def reset(self,URDF_PATH):
        builder = warp.sim.ModelBuilder(up_vector=(0.0, 0.0, 1.0))
        
        # 不直接使用add_shape_box : box过于宽扁，碰撞检测会出现问题
        builder.add_shape_plane(pos=self.board_initial_position,rot=wp.quat_from_axis_angle(wp.vec3((1.,0.,0.)), wp.float32(np.pi/2)),width=self.shuffboard_size[0],length=self.shuffboard_size[1],ke=self.ke, kf=self.kf, kd=self.kd, mu=self.mu)

        b = builder.add_body(origin=warp.transform([0.0,0.0,0.0], warp.quat_identity()))
        builder.add_shape_box(body=b, hx=self.box_size[0], hy=self.box_size[1], hz=self.box_size[2], density=750, ke=self.ke, kf=self.kf, kd=self.kd, mu=self.mu)
        builder.add_joint_free(child=b,parent_xform=warp.transform(self.initial_position, warp.quat_identity()), name="free_joint")
        
        warp.sim.parse_urdf(
            URDF_PATH,
            builder,
            floating=False,
            enable_self_collisions = True
        )

        # use `requires_grad=True` to create a model for differentiable simulation
        self.model = builder.finalize()

        self.model.ground = True

        self.model.soft_contact_ke = self.ke
        self.model.soft_contact_kf = self.kf
        self.model.soft_contact_kd = self.kd
        self.model.soft_contact_mu = self.mu
        self.model.soft_contact_margin = 1.0
        self.model.soft_contact_restitution = 1.0

    def simple_model(self):
        self.ke_simple = 50
        self.kd_simple = 0.5
        builder = warp.sim.ModelBuilder(up_vector=(0.0, 0.0, 1.0))
        builder.add_shape_plane(pos=self.board_initial_position,rot=wp.quat_from_axis_angle(wp.vec3((1.,0.,0.)), wp.float32(np.pi/2)),width=self.shuffboard_size[0],length=self.shuffboard_size[1],ke=self.ke_simple, kf=self.kf, kd=self.kd_simple, mu=self.mu)

        b = builder.add_body(origin=warp.transform([0.0,0.0,0.0], warp.quat_identity()))
        builder.add_shape_box(body=b, hx=self.box_size[0], hy=self.box_size[1], hz=self.box_size[2], density=750, ke=self.ke_simple, kf=self.kf, kd=self.kd_simple, mu=self.mu)
        builder.add_joint_free(child=b,parent_xform=warp.transform(self.initial_position, warp.quat_identity()), name="free_joint")

        model_simple = builder.finalize()

        model_simple.ground = True

        model_simple.soft_contact_ke = self.ke
        model_simple.soft_contact_kf = self.kf
        model_simple.soft_contact_kd = self.kd
        model_simple.soft_contact_mu = self.mu
        model_simple.soft_contact_margin = 1.0
        model_simple.soft_contact_restitution = 1.0

        model_simple.gravity = wp.array(model_simple.gravity.reshape(1, 3),dtype=wp.vec3)
        # joint_q[:3] = joint_q[:3] + self.initial_position
        # model_simple.body_q.assign(joint_q)
        # model_simple.body_qd.assign(joint_qd)

        return model_simple


    def simulate_slow(self,ee_speed):
        for j in range(self.sim_substeps):
            control = self.model.control()
            self.joint_qd_franka = compute_joint_v_from_ee(self.state_sequence[0].joint_q.numpy()[7:],self.orientation*ee_speed)
            control.joint_act.assign(self.joint_qd_franka)
            
            self.state_sequence[0].clear_forces()
            warp.sim.collide(self.model, self.state_sequence[0])
            self.integrator.simulate(
                self.model,
                self.state_sequence[0],
                self.state_sequence[1],
                self.sim_dt,
                control=control,
            )
            # 夹爪强制归位
            self.state_sequence[0].joint_q[-2:].fill_(0.0)
            self.state_sequence[0].joint_qd[-2:].fill_(0.0)
            self.state_sequence[1].joint_q[-2:].fill_(0.0)
            self.state_sequence[1].joint_qd[-2:].fill_(0.0)

            # if np.linalg.norm(self.state_sequence[1].joint_qd.numpy()[3:6])>0.1 and self.flag:
            #     print('After hitting time:')
            #     print(f'Box linear speed: {np.linalg.norm(self.state_sequence[1].joint_qd.numpy()[3:6])} ')
            #     print(f'xyz: {self.state_sequence[1].joint_qd.numpy()[3:6]}')
                # print(f'End effector speed: {np.linalg.norm(self.state_sequence[1].body_qd.numpy()[10,3:6])} ')
                # print(f'xyz: {self.state_sequence[1].body_qd.numpy()[10,3:6]}')
                # self.flag = 0
            
            self.state_sequence[0], self.state_sequence[1] = self.state_sequence[1], self.state_sequence[0]

    def simulate_fast(self):
        for j in range(self.sim_substeps):
            self.state_sequence[0].clear_forces()
            warp.sim.collide(self.model_simple, self.state_sequence[0])
            self.integrator_fast.simulate(
                self.model_simple,
                self.state_sequence[0],
                self.state_sequence[1],
                self.sim_dt,
            )
            
            self.state_sequence[0], self.state_sequence[1] = self.state_sequence[1], self.state_sequence[0]


    def forward(self,ee_speed):
        error = 10.0

        with warp.ScopedTimer("slow step"):
            # slow simulation
            self.state_sequence = [self.model.state(), self.model.state()]
            self.state_sequence[0].joint_q.assign(self.model.joint_q.numpy().tolist()[:7] +self.joint_q_franka)
            warp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_sequence[0])
            self.flag = 0
            self.model.joint_axis_mode.assign([2]*9)
            self.model.joint_target_ke.fill_(20.0)
            # if self.renderer:
            #     self.render()

            self.sim_substeps = self.sim_substeps_1
            self.sim_dt = self.sim_dt_1
            for i in range(int(6/ee_speed)):
                self.simulate_slow(ee_speed)
                # if self.renderer:
                #     self.render()

                box_linear_speed = np.linalg.norm(self.state_sequence[0].joint_qd.numpy()[3:6])
                if box_linear_speed > 0.1 and self.flag==0:
                    self.flag = 1
        
        with warp.ScopedTimer("fast step"):
            # fast simulation
            self.sim_substeps = self.sim_substeps_2
            self.sim_dt = self.sim_dt_2

            # integrator = warp.sim.SemiImplicitIntegrator()
            self.model_simple.joint_q.assign(self.state_sequence[0].joint_q.numpy()[:7] )
            self.model_simple.joint_qd.assign(self.state_sequence[0].joint_qd.numpy()[:6] )
            self.state_sequence = [self.model_simple.state(), self.model_simple.state()]
            self.state_sequence[0].joint_q.assign(self.model_simple.joint_q.numpy().tolist())

            warp.sim.eval_fk(self.model_simple, self.model_simple.joint_q, self.model_simple.joint_qd, None, self.state_sequence[0])


            for i in range(int(6/ee_speed),self.num_frames):
                self.simulate_fast()            
                
                self.render(renderer=self.renderer)
                
                box_linear_speed = np.linalg.norm(self.state_sequence[0].joint_qd.numpy()[3:6]) 
                if box_linear_speed < 0.01 and self.flag==1:
                    projection = np.dot(self.state_sequence[0].joint_q.numpy()[:3],self.orientation)
                    # projection = np.dot((self.state_sequence[0].body_q.numpy()[0,:3]-self.initial_position),self.orientation)
                    error = projection - np.dot(np.array(self.target)-np.array(self.initial_position),self.orientation)
                    # pdb.set_trace()
                    break

        if self.flag == 0:
            print('Not hitting the box')
        elif error == 10.0:
            print('Need more time for simulation')
        return error

    def step(self,ee_speed):
        with warp.ScopedTimer("step"):
            error = self.forward(ee_speed)
        
        return error,self.joint_q_franka


    def render(self,renderer):
            renderer.begin_frame(self.render_time)
            renderer.render(self.state_sequence[0])
            renderer.render_box(
                pos=self.target,
                rot=warp.quat_identity(),
                extents=self.box_size,
                name="target",
                color=(1.0, 0.0, 0.0),
            )

            renderer.end_frame()
            self.render_time += self.frame_dt


# env = WarpFrankaEnv(stage_path='franka_shuffleboard.usda',integrator='featherstone',num_frames = 120, mu = 0.05)
# env.step()

# if env.renderer:
#     env.renderer.save()