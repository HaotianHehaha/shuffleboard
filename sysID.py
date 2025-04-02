
import numpy as np
import warp
import trajectory_tracking
import warp as wp
import warp.optim
import warp.render
import warp.sim
import warp.sim.render
import warp.types
import os
import pdb

width1 = 0.04
width2 = 0.04
height = 0.04

mass = 0.0488
density = mass / (width1 * width2 * height)

@wp.kernel
def assign_param(params: wp.array(dtype=wp.float32), shape_materials: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    shape_materials[tid] = params[tid]

@wp.kernel
def loss_kernel(body_q: wp.array(dtype=wp.transform), target:  wp.vec3, loss: wp.array(dtype=float)):

    x = wp.transform_get_translation(body_q[0])
    # distance to target
    delta = x- target
    loss[0] = wp.dot(delta , delta)

@wp.kernel
def step_kernel(x: wp.array(dtype=float), grad: wp.array(dtype=float), alpha: float):
    tid = wp.tid()

    if grad[tid] > 0:
        x[tid] = x[tid] * (1.0 - alpha)
    elif grad[tid] < 0:
        x[tid] = x[tid] * (1.0 + alpha)

@wp.kernel
def enforce_grad_kernel(lower_bound: wp.float32, upper_bound: wp.float32, x: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    if x[tid] < lower_bound:
        x[tid] = lower_bound
    elif x[tid] > upper_bound:
        x[tid] = upper_bound
    elif wp.isnan(x[tid]):
        x[tid] = 0.0


class sysID():
    def __init__(self,  filepath = 'apriltag_poses_1741847062.json', verbose=False):
        self.verbose = verbose

        # get real robot trajectory
        self.real_setting = trajectory_tracking.tracking(filepath)
        # seconds
        sim_duration = 3.0 #(self.real_setting['final_time'] - self.real_setting['initial_time'])/1000

        # control frequency
        fps = 60
        self.frame_dt =1.0 / fps
        frame_steps = int(sim_duration / self.frame_dt)

        # sim frequency
        self.sim_substeps = 8
        self.sim_steps = frame_steps * self.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iter = 0
        self.render_time = 0.0

        self.train_rate = 3.0e-3

        self.ke = 100
        self.kd = 1
        self.kf = 100.0
        self.mu = 0.03
        self.grad_lower_bound = wp.float32(-1)
        self.grad_upper_bound = wp.float32(1)

        self.shuffboard_size = [2.0, 2.0 ]
        self.board_initial_position = [0.0, 0.0, 0.065]

        self.initial_position = self.real_setting['initial_position']
        self.initial_position[-1] = height/2 + self.board_initial_position[-1]

        builder = warp.sim.ModelBuilder(up_vector=(0.0, 0.0, 1.0))

        b = builder.add_body(origin=wp.transform(self.initial_position, self.real_setting['initial_quaternion']))
        builder.add_shape_box(body=b, hx=width1/2, hy=width2/2, hz=height/2, density=density,ke=self.ke, kf=self.kf, kd=self.kd, mu=self.mu)

        builder.add_shape_plane(pos=self.board_initial_position,rot=wp.quat_from_axis_angle(wp.vec3((1.,0.,0.)), wp.float32(np.pi/2)),width=self.shuffboard_size[0],length=self.shuffboard_size[1],ke=self.ke, kf=self.kf, kd=self.kd, mu=self.mu)

        # use `requires_grad=True` to create a model for differentiable simulation
        self.model = builder.finalize(requires_grad=True)
        self.model.ground = True

        self.model.soft_contact_ke = self.ke
        self.model.soft_contact_kf = self.kf
        self.model.soft_contact_kd = self.kd
        self.model.soft_contact_mu = self.mu
        self.model.soft_contact_margin = 1.0
        self.model.soft_contact_restitution = 1.0

        self.model.body_qd.assign(
                [
                    [0.0, 0.0, 0.0]+self.real_setting['initial_velocity'],
                ]
            )

        self.integrator = warp.sim.SemiImplicitIntegrator()

        self.target = self.real_setting['final_position']
        self.target[-1] = height/2 + self.board_initial_position[-1]
       
        self.loss = wp.zeros(1, dtype=float,requires_grad=True)

        # allocate sim states for trajectory
        self.states = []
        for _ in range(self.sim_steps + 1):
            self.states.append(self.model.state())
        

        self.optimizer = warp.optim.SGD(
            [self.model.shape_materials.mu],
            lr=self.train_rate,
        )

        self.renderer = warp.sim.render.SimRenderer(model=self.model, path='box.usda', scaling=1.0)

        # capture forward/backward passes
        self.use_cuda_graph = False#wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.tape = wp.Tape()
                with self.tape:
                    self.forward()
                self.tape.backward(self.loss)
            self.graph = capture.graph

    def forward(self):
        # run control loop
        # print(self.states[0].body_q.numpy()[0,:3])
        # print(self.states[0].body_qd.numpy()[0,3:])
        for i in range(self.sim_steps):
            warp.sim.collide(self.model, self.states[i])
            self.states[i].clear_forces()
            self.integrator.simulate(self.model, self.states[i], self.states[i + 1], self.sim_dt)

        # print(self.states[-1].body_q.numpy()[0,:3])
        # print(self.states[-1].body_qd.numpy()[0,3:])
        wp.launch(loss_kernel, dim=1, inputs=[self.states[-1].body_q, wp.vec3(self.target), self.loss])
        return self.loss

    def step(self):
        # with wp.ScopedTimer("step"):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.tape = wp.Tape()
                with self.tape:
                    self.forward()
                self.tape.backward(self.loss)

            # gradient descent step
            x = self.model.shape_materials.mu

            # self.render()
            # self.renderer.save()
            if self.loss.numpy()[0]< 1e-4:
                flag = 1
            else:
                flag = 0

            # pdb.set_trace()
            wp.launch(enforce_grad_kernel, dim=x.grad.shape[0], inputs=[self.grad_lower_bound, self.grad_upper_bound, x.grad])

            if self.verbose:
                print(f"Iter: {self.iter} Loss: {self.loss}")
                print(f"mu: {x.numpy()} g: {x.grad.numpy()}")

            self.optimizer.step([x.grad])
            # clear grads for next iteration
            self.tape.zero()

            # set same mu for all shapes
            self.model.shape_materials.mu.assign(x.numpy()[0]*np.ones(x.shape[0]))

            self.iter = self.iter + 1

            return flag

    def render(self):

        with wp.ScopedTimer("render"):
            for i in range(0, self.sim_steps, self.sim_substeps):

                self.renderer.begin_frame(self.render_time)
                self.renderer.render(self.states[i])
                self.renderer.render_box(
                    pos=self.target,
                    rot=self.real_setting['final_quaternion'],
                    extents=(width1/2, width2/2, height/2),
                    name="target",
                    color=(0.0, 0.0, 0.0),
                )

                self.renderer.end_frame()
                self.render_time += self.frame_dt

    def check_grad(self):
        param = self.states[0].particle_qd

        # initial value
        x_c = param.numpy().flatten()

        # compute numeric gradient
        x_grad_numeric = np.zeros_like(x_c)

        for i in range(len(x_c)):
            eps = 1.0e-3

            step = np.zeros_like(x_c)
            step[i] = eps

            x_1 = x_c + step
            x_0 = x_c - step

            param.assign(x_1)
            l_1 = self.forward().numpy()[0]

            param.assign(x_0)
            l_0 = self.forward().numpy()[0]

            dldx = (l_1 - l_0) / (eps * 2.0)

            x_grad_numeric[i] = dldx

        # reset initial state
        param.assign(x_c)

        # compute analytic gradient
        tape = wp.Tape()
        with tape:
            l = self.forward()

        tape.backward(l)

        x_grad_analytic = tape.gradients[param]

        print(f"numeric grad: {x_grad_numeric}")
        print(f"analytic grad: {x_grad_analytic}")

        tape.zero()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="box.usda",
        help="Path to the output USD file.",
    )
    parser.add_argument(
        "--filepath",
        type=lambda x: None if x == "None" else str(x),
        default="raw_data/apriltag_poses_1741847062.json",
        help="Path to origin data file.",
    )
    parser.add_argument("--train_iters", type=int, default=30, help="Total number of training iterations.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        # 读取raw_data文件夹内所有json文件
        for file in os.listdir('raw_data'):
            if file.endswith(".json"):
                args.filepath = os.path.join('raw_data', file)
                example = sysID( verbose=args.verbose,filepath=args.filepath)

                # replay and optimize
                for i in range(args.train_iters):
                    example.step()
                
                # print(f'{args.filepath}: mu: {example.model.shape_materials.mu.numpy()}')
                    if i % 10 == 0 or i == args.train_iters-1:
                        example.render()

                if example.renderer:
                    example.renderer.save()

                break


# python sim_vs_real.py --stage_path box_mu.usda --verbose




