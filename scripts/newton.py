
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import  MultiThreadedExecutor
from rclpy.node import Node
from ament_index_python import get_package_share_directory

from unitree_go.msg import LowCmd, LowState, IMUState, MotorState

import gc
import numpy as np
import os
import threading
import time

import newton
import newton.examples
import warp as wp

Q_OFFSET = 7
QD_OFFSET = 6
N_MOTORS = 12
GRAVITY = 9.81


class NodeNewton(Node):
    """
    """
    def __init__(self, viewer) -> None:
        """
        """
        super().__init__("node_newton_physics")

        self.declare_parameter("newton_sim_hz", 500)
        self.declare_parameter("newton_sim_substeps", 2) # must be even
        self.declare_parameter("newton_render_interval", 10)
        self.declare_parameter("newton_use_mujoco_contacts", True)
        self.declare_parameter("quadruped_mjcf_pkg", "qdesc_go2")

        self.sim_hz = self.get_parameter("newton_sim_hz").get_parameter_value().integer_value
        self.render_interval = self.get_parameter("newton_render_interval").get_parameter_value().integer_value
        self.fps = self.sim_hz / self.render_interval

        self.frame_dt = 1.0 / self.sim_hz
        self.sim_time = 0.0

        self.frame_index = 0

        self.sim_substeps = self.get_parameter("newton_sim_substeps").get_parameter_value().integer_value
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.device = wp.get_device()
        self.viewer = viewer

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder=builder)

        mjcf_pkg = self.get_parameter("quadruped_mjcf_pkg").get_parameter_value().string_value
        mjcf_path = os.path.join(get_package_share_directory(mjcf_pkg), "description", "quadruped.xml")
        builder.add_mjcf(
            source=mjcf_path,
            floating=True,
            enable_self_collisions=False
        )
        builder.add_ground_plane()

        self.model = builder.finalize(device=self.device)
        self.viewer.set_model(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(model=self.model, joint_q=self.state_0.joint_q, joint_qd=self.state_0.joint_qd, state=self.state_0)

        self.use_mujoco_contacts = self.get_parameter("newton_use_mujoco_contacts").get_parameter_value().bool_value
        self.solver = newton.solvers.SolverMuJoCo(
            model=self.model,
            use_mujoco_contacts=self.use_mujoco_contacts,
            njmax=300,
            nconmax=100
        )

        self.model.joint_target_ke.fill_(0.0)
        self.model.joint_target_kd.fill_(0.0)
        self.solver.notify_model_changed(newton.solvers.SolverNotifyFlags.JOINT_DOF_PROPERTIES)

        if self.use_mujoco_contacts:
            self.contacts = newton.Contacts(  
                rigid_contact_max=self.solver.mjw_data.naconmax,  
                soft_contact_max=0,  
                device=self.device,  
            )
        else:
            self.contacts = self.model.contacts()

        dof = self.model.joint_dof_count
        self._target_torque_wp = wp.zeros(shape=dof, dtype=wp.float32, device=self.device)
        self._input_torque_np = np.zeros(dof, dtype=np.float32)

        if self.control.joint_f is None:
            self.control.joint_f = wp.zeros(shape=dof, dtype=wp.float32, device=self.device)

        self.ctrlrange = self.model.mujoco.actuator_ctrlrange.numpy()  

        self._prev_q  = np.zeros(N_MOTORS, dtype=np.float32)
        self._prev_dq = np.zeros(N_MOTORS, dtype=np.float32)

        self._lock = threading.Lock()
        self.graph = None
        self.capture()

        self._last_cmd_time: float = 0.0
        self.CMD_TIMEOUT_SEC: float = 0.1

        self.pub_lowstate_ = self.create_publisher(
            msg_type=LowState,
            topic="lowstate",
            qos_profile=10
        )

        self.sub_lowstate_ = self.create_subscription(
            msg_type=LowCmd,
            topic="lowcmd",
            callback=self.handle_subscription_lowcmd_,
            qos_profile=10
        )


    def capture(self) -> None:
        """
        """
        if not self.device.is_cuda:
            return
        self.get_logger().info("Capturing CUDA graph...")
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph
        self.get_logger().info("Captured CUDA graph.")


    def simulate(self) -> None:
        """
        """
        if not self.use_mujoco_contacts:
            self.model.collide(state=self.state_0, contacts=self.contacts)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        
        if self.use_mujoco_contacts:
            self.solver.update_contacts(contacts=self.contacts, state=self.state_0)


    def step(self) -> None:
        """
        """
        if time.perf_counter() - self._last_cmd_time > self.CMD_TIMEOUT_SEC:
            with self._lock:
                self._input_torque_np[:] = 0.0
                self._target_torque_wp.assign(self._input_torque_np)

        with self._lock:
            wp.copy(dest=self.control.joint_f, src=self._target_torque_wp)

            if self.graph:
                wp.capture_launch(graph=self.graph)
            else:
                self.simulate()

            if self.viewer and (self.frame_index % self.render_interval == 0):
                self.viewer.begin_frame(self.sim_time)
                self.viewer.log_state(self.state_0)
                self.viewer.end_frame()
            
            wp.synchronize()

            self.sim_time += self.frame_dt
            self.frame_index += 1
        
        self.handle_publisher_lowstate_()


    def wrap(self, angle: float) -> float:
        """
        """
        return angle
        # return (angle + np.pi) % (2*np.pi) - np.pi


    @staticmethod
    def _quat_to_rpy(w: float, x: float, y: float, z: float) -> tuple:
        """
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw
    

    def _estimate_foot_forces(self) -> np.ndarray:
        """
        Estimate normal contact forces at each foot (FL, FR, RL, RR) in Newtons.

        When using MuJoCo contacts the solver exposes contact forces; we sum
        the normal component for contacts associated with each foot geom.
        Falls back to zeros if contact data isn't accessible.
        """
        forces = np.zeros(4, dtype=np.float32)  # [FL, FR, RL, RR]

        try:
            if self.use_mujoco_contacts:
                # MuJoCo contact forces are in self.solver.mjw_data
                # mjw_data.cfrc_ext shape: (nbody, 6) — spatial force on each body
                # Foot body indices depend on MJCF parse order; use geom names as proxy.
                # Simpler: use efc_force (constraint forces) filtered by contact type.
                # The most accessible route: mjw_data.sensordata for touch sensors.
                # MJCF defines touch sensors: FL_touch[0], FR_touch[1], RL_touch[2], RR_touch[3]
                sdata = self.solver.mjw_data.sensordata  # numpy array
                if sdata is not None and len(sdata) >= 4:
                    # Touch sensor indices in MJCF sensor block:
                    # sensors are ordered: 12 jointpos + 12 jointvel + 12 jointactuatorfrc
                    #                      + FL_touch, FR_touch, RL_touch, RR_touch = indices 36..39
                    TOUCH_OFFSET = 36
                    forces[0] = max(0.0, float(sdata[TOUCH_OFFSET + 0]))  # FL
                    forces[1] = max(0.0, float(sdata[TOUCH_OFFSET + 1]))  # FR
                    forces[2] = max(0.0, float(sdata[TOUCH_OFFSET + 2]))  # RL
                    forces[3] = max(0.0, float(sdata[TOUCH_OFFSET + 3]))  # RR
            else:
                # For the Warp contact pipeline we don't have easy per-foot normals;
                # return zeros (or you could integrate contact_f from self.contacts).
                pass
        except Exception:
            pass  # non-fatal — foot forces remain zero

        return forces


    def handle_subscription_lowcmd_(self, msg: LowCmd) -> None:
        """
        """
        # SDK_TO_NEWTON_DOFS = [9, 10, 11, 6, 7, 8, 15, 16, 17, N_MOTORS, 13, 14]
        self._last_cmd_time = time.perf_counter()
        with self._lock:
            q_current = self.state_0.joint_q.numpy()
            qd_current = self.state_0.joint_qd.numpy()
            for i in range(N_MOTORS):
                j1 = i + Q_OFFSET
                j2 = i + QD_OFFSET
                k = j2 # SDK_TO_NEWTON_DOFS[i]
                motor_cmd = msg.motor_cmd[i]
                tau_pd = motor_cmd.kp * self.wrap(motor_cmd.q - q_current[j1]) + motor_cmd.kd * self.wrap(motor_cmd.dq - qd_current[j2])
                tau = tau_pd + motor_cmd.tau
                tau = np.clip(tau, self.ctrlrange[i, 0], self.ctrlrange[i, 1])
                
                self._input_torque_np[k] = tau
            self._target_torque_wp.assign(self._input_torque_np)


    def handle_publisher_lowstate_(self) -> None:
        """
        """
        with self._lock:
            q_ = self.state_0.joint_q.numpy()
            qd_ = self.state_0.joint_qd.numpy()

        msg = LowState()

        quat = q_[3:7]
        gyro = qd_[3:6]

        w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        gx =  2.0 * (x*z - w*y) * GRAVITY
        gy =  2.0 * (y*z + w*x) * GRAVITY
        gz = (w*w - x*x - y*y + z*z) * GRAVITY

        imu = IMUState()
        imu.quaternion = [w, x, y, z]
        imu.gyroscope = [float(gyro[0]), float(gyro[1]), float(gyro[2])]
        imu.accelerometer = [float(gx), float(gy), float(gz)]
        imu.rpy = list(self._quat_to_rpy(w, x, y, z))

        msg.imu_state = imu

        q_joints = q_[Q_OFFSET:Q_OFFSET+N_MOTORS].astype(np.float32)
        qd_joints = qd_[QD_OFFSET:QD_OFFSET+N_MOTORS].astype(np.float32)
        qdd_joints = (qd_joints - self._prev_dq) / self.frame_dt

        for i in range(N_MOTORS):
            ms = MotorState()
            ms.mode  = 0x01
            ms.q     = float(q_joints[i])
            ms.dq    = float(qd_joints[i])
            ms.ddq   = float(qdd_joints[i])
            ms.tau_est = float(self._input_torque_np[QD_OFFSET + i])
            msg.motor_state[i] = ms

        self._prev_q  = q_joints.copy()
        self._prev_dq = qd_joints.copy()

        foot_forces = self._estimate_foot_forces()
        msg.foot_force_est[0] = int(foot_forces[0])  # FL
        msg.foot_force_est[1] = int(foot_forces[1])  # FR
        msg.foot_force_est[2] = int(foot_forces[2])  # RL
        msg.foot_force_est[3] = int(foot_forces[3])  # RR

        self.pub_lowstate_.publish(msg)


class NodeNewtonWrapper():
    """
    """
    def __init__(self, node: NodeNewton, executor: MultiThreadedExecutor) -> None:
        """
        """
        self.node = node
        self.executor = executor


    def warm(self, n_steps: int = 20) -> None:
        """
        """
        self.node.get_logger().info(f"")
        self.node.get_logger().info(f"Warming up Newton ({n_steps} steps)...")
        for i in range(n_steps):
            self.node.step()
            progress = "#" * (i + 1) + "-" * (n_steps - i - 1)
            self.node.get_logger().info(f"{progress} {i + 1}")
        self.node.get_logger().info(f"Warmed up Newton.")


    def run(self) -> None:
        """
        """
        gc.disable()
        try:
            while rclpy.ok():
                t_0 = time.perf_counter()
                self.executor.spin_once(timeout_sec=0)
                self.node.step()
                
                dt = time.perf_counter() - t_0
                t_sleep = self.node.frame_dt - dt
                if t_sleep > 0:
                    time.sleep(t_sleep)
                gc.collect(0)
        except KeyboardInterrupt:
            pass
        finally:
            gc.enable()


def main(args=None) -> None:
    """
    """
    viewer, _ = newton.examples.init()

    rclpy.init(args=args)
    node = NodeNewton(viewer=viewer)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node=node)
    wrapper = NodeNewtonWrapper(node=node, executor=executor)

    try:
        wrapper.warm()
        wrapper.run()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, shutting down...")
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
