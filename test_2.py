
import warp as wp
import numpy as np

import warp.sim
import warp.sim.render
import pdb

@wp.kernel
def loss_kernel(body_q: wp.array(dtype=wp.transform), target:  wp.vec3, loss: wp.array(dtype=float)):

    x = wp.transform_get_translation(body_q[-1])
    # distance to target
    delta = x- target
    loss[0] = wp.dot(delta , delta)


class WarpFrankaEnv():
    def __init__(self,stage_path='franka_shuffleboard.usda',integrator='featherstone',num_frames = 60):

        self.num_frames = num_frames
        # control frequency
        fps = 60
        self.frame_dt = 1.0 / fps

        # sim frequency
        self.sim_substeps = 30
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iter = 0
        self.render_time = 0.0
        self.ke = 1e4
        self.kd = 10.0
        self.kf = 100.0
        self.mu = 0.02

        self.shuffboard_size = [0.6, 0.15 ]
        self.board_initial_position = [1.2, 0.0, 0.065]
        self.box_size = [0.015,0.015,0.015]
        self.initial_position = [0.5,0.0,0.055]

        self.target = [1.0, 0.0,0.055]
        self.loss = wp.zeros(1, dtype=float,requires_grad=True)     

        self.reset()

        if integrator == 'featherstone':
            self.integrator = warp.sim.FeatherstoneIntegrator(self.model)
        elif integrator == 'semi-implicit':
            self.integrator = warp.sim.SemiImplicitIntegrator()
        elif integrator == 'xpbd':
            self.integrator = warp.sim.XPBDIntegrator(iterations=2)
        else:
            raise ValueError(f"Unknown integrator: {integrator}")
        
        # set ground parameters
        self.model.shape_materials.kd.fill_(self.kd)
        self.model.shape_materials.ke[-1:].fill_(self.ke)
        # self.model.shape_materials.ke.fill_(self.ke)
        self.model.shape_materials.kf[-1:].fill_(self.kf)
        self.model.shape_materials.mu.fill_(self.mu)
        print(self.model.shape_materials)
        # self.model.rigid_contact_margin = 1e-3
        self.model.gravity = wp.array(self.model.gravity,dtype=wp.vec3)

        self.state_sequence = [self.model.state()]
        # [0.0,0.0,0.0,1.1,1.1,0.0]
        
        # warp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_sequence[0])
        
        self.state_sequence.append(self.model.state())

        self.state_sequence[0].joint_qd.assign([0.0,0.0,0.0,0.5,0.0,0.0])
        # self.state_sequence[0].body_qd.assign([0.0,0.0,0.0,0.3,0.0,0.0])

        if stage_path:
            self.renderer = warp.sim.render.SimRenderer(self.model, stage_path, scaling=1.0)
    
    def reset(self):
        builder = warp.sim.ModelBuilder(up_vector=(0., 0., 1.))
        
        # block_width = 0.02
        # hx_num = self.shuffboard_size[0]/block_width
        # hy_num = self.shuffboard_size[1]/block_width
        # initial_x = self.board_initial_position[0]-self.shuffboard_size[0]+block_width
        # final_x = self.board_initial_position[0]+self.shuffboard_size[0]-block_width
        # x_sequence = np.linspace(initial_x,final_x,int(hx_num))
        # initial_y = self.board_initial_position[1]-self.shuffboard_size[1]+block_width
        # final_y = self.board_initial_position[1]+self.shuffboard_size[1]-block_width
        # y_sequence = np.linspace(initial_y,final_y,int(hy_num))
        # X, Y, Z= np.meshgrid(x_sequence,y_sequence,np.array([self.shuffboard_size[2]]))
        # mesh_points_3d = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
        # for center in mesh_points_3d:
        #     builder.add_shape_box(body=-1, pos=wp.vec3(center),hx=block_width, hy=block_width, hz=self.shuffboard_size[2], density=3000, ke=self.ke, kf=self.kf, kd=self.kd, mu=self.mu)
        builder.add_shape_plane(pos=(1.2, 0.,0.04),rot=wp.quat_from_axis_angle(wp.vec3((1.,0.,0.)), wp.float32(np.pi/2)),width=1.0,length=0.2,ke=self.ke, kf=self.kf, kd=self.kd, mu=self.mu)

        b = builder.add_body(origin=wp.transform(self.initial_position, wp.quat_identity()))
        builder.add_shape_box(body=b,hx=self.box_size[0], hy=self.box_size[1], hz=self.box_size[2], density=750, ke=self.ke, kf=self.kf, kd=self.kd, mu=self.mu)
        builder.add_joint_free(child=b, parent_xform = wp.transform(self.initial_position, wp.quat_identity()),name="free_joint")


        # use `requires_grad=True` to create a model for differentiable simulation
        self.model = builder.finalize()

        self.model.ground = True

        self.model.soft_contact_ke = self.ke
        self.model.soft_contact_kf = self.kf
        self.model.soft_contact_kd = self.kd
        self.model.soft_contact_mu = self.mu
        self.model.soft_contact_margin = 1.0
        self.model.soft_contact_restitution = 1.0

        # print(self.model.shape_collision_radius)
        # self.model.shape_collision_radius[:1].fill_(1e-2)

    
    def forward(self):
        self.render()
        for i in range(self.num_frames):
            for j in range(self.sim_substeps):
                self.state_sequence[0].clear_forces()
                warp.sim.collide(self.model, self.state_sequence[0])
                
                self.integrator.simulate(
                    self.model,
                    self.state_sequence[0],
                    self.state_sequence[1],
                    self.sim_dt,
                )
                # if i + j ==0:
                #     print(self.state_sequence[0].body_q)
                #     print(self.model.rigid_contact_count)
                #     # print(self.model.rigid_contact_point0)
                #     print(f'body force:{self.state_sequence[0].body_f}')
                self.state_sequence[0], self.state_sequence[1] = self.state_sequence[1], self.state_sequence[0]

                # print(self.state_sequence[0].joint_q.numpy()[2])
                # print(self.state_sequence[1].body_qd.numpy()[0])
                # pdb.set_trace()
            self.render()
                # pdb.set_trace()

        wp.launch(loss_kernel, dim=1, inputs=[self.state_sequence[0].body_q, wp.vec3(self.target)], outputs=[self.loss])
        return self.loss

    def step(self):
        with warp.ScopedTimer("step"):
            self.forward()


    def render(self):
        # with warp.ScopedTimer("render"):
            self.renderer.begin_frame(self.render_time)
            self.renderer.render(self.state_sequence[0])
            self.renderer.end_frame()
            self.render_time += self.frame_dt


env = WarpFrankaEnv(stage_path='franka_shuffleboard.usda',integrator='featherstone',num_frames = 300)
# featherstone semi-implicit
env.step()

if env.renderer:
    env.renderer.save()