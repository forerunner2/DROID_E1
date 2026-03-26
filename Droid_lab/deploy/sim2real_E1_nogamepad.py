import argparse
import math
import time

import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from base.LegBase import LegBase
from base.Base import get_command
from base.Base import NanoSleep, euler_to_quaternion, quat_rotate_inverse
from tools.Gamepad import GamepadHandler
from tools.CircularBuffer import CircularBuffer
from tools.load_env_config import load_configuration


onnx_mode_path = "policies/policy.onnx"

IsaacLabJointOrder = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]
RealJointOrder = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
Isaac_to_Real_indices = [IsaacLabJointOrder.index(joint) for joint in RealJointOrder]
Real_to_Isaac_indices = [RealJointOrder.index(joint) for joint in IsaacLabJointOrder]


class DummyRC:
    class State:
        START = True
        LT = 0
        LEFT_Y = 0.0
        RIGHT_X = 0.0
        LEFT_X = 0.0

    state = State()


class Sim2Real(LegBase):
    def __init__(self, use_gamepad: bool = True):
        LegBase.__init__(self)
        self.num_actions = 12
        self.num_observations = 47
        self.gait_frequency = 0
        self.cfg = load_configuration("policies/env_cfg.json", RealJointOrder)
        self.run_flag = True
        self.use_gamepad = use_gamepad

        self.command = [0.0, 0.0, 0.0]
        self.target_q = np.zeros(self.num_actions, dtype=np.double)
        self.action = np.zeros(self.num_actions, dtype=np.double)
        self.onnx_policy = ort.InferenceSession(onnx_mode_path)
        self.hist_obs = CircularBuffer(self.num_observations, self.cfg.hist_length)
        self.rc = GamepadHandler() if use_gamepad else DummyRC()

    def init_robot(self):
        print("default_joints:", self.cfg.default_joints)
        init_pos = np.append(self.cfg.default_joints, 0.0)
        self.set_leg_path(1, init_pos)
        if not self.use_gamepad:
            print("No gamepad: auto-start")
            return

        timer = NanoSleep(self.cfg.decimation)
        print("Press START to run, hold LT for emergency stop")
        while (self.rc.state.START is False) and (self.run_flag is True):
            start_time = time.perf_counter()
            self.get_leg_state()
            if self.rc.state.LT > 64:
                print("Emergency stop")
                exit()
            timer.waiting(start_time)

    def update_rc_command(self):
        if not self.use_gamepad:
            self.command = [0.0, 0.0, 0.0]
            # self.gait_frequency = 0
            self.gait_frequency = 1.5
            return

        self.command[0] = get_command(self.command[0], self.rc.state.LEFT_Y * 1.0, 0.05)
        self.command[1] = get_command(self.command[1], self.rc.state.RIGHT_X * 1.0, 0.05)
        self.command[2] = get_command(self.command[2], self.rc.state.LEFT_X * 1.0, 0.05)
        self.gait_frequency = 1.5

    def get_gravity_orientation_from_rpy(self, roll, pitch):
        rot = R.from_euler("xy", [roll, pitch])
        g_world = np.array([0, 0, -1])
        g_local = rot.inv().apply(g_world)
        return g_local

    def get_obs(self, gait_process):
        q = np.array(self.legState.position[:12])
        dq = np.array(self.legState.velocity[:12])

        base_euler = np.array(self.legState.imu_euler)
        base_ang_vel = np.array(self.legState.imu_gyro)

        base_euler[base_euler > math.pi] -= 2 * math.pi
        eq = euler_to_quaternion(base_euler[0], base_euler[1], base_euler[2])
        eq = np.array(eq, dtype=np.double)
        project_gravity = quat_rotate_inverse(eq, np.array([0.0, 0.0, -1]))

        self.update_rc_command()

        obs = np.zeros([self.num_observations], dtype=np.float32)
        obs[0:3] = base_ang_vel
        obs[3:6] = project_gravity
        obs[6:9] = self.command
        obs[9] = np.cos(2 * np.pi * gait_process) * (self.gait_frequency > 1.0e-8)
        obs[10] = np.sin(2 * np.pi * gait_process) * (self.gait_frequency > 1.0e-8)
        obs[11:23] = (q - self.cfg.default_joints)[Real_to_Isaac_indices]
        obs[23:35] = dq[Real_to_Isaac_indices]
        obs[35:47] = self.action[Real_to_Isaac_indices]
        obs = np.clip(obs, -100, 100)
        return q, dq, obs

    def get_action(self, obs):
        obs = [np.array(obs, dtype=np.float32)]
        action = np.array(self.onnx_policy.run(None, {"obs": obs})[0].tolist()[0])
        self.action = np.clip(action[Isaac_to_Real_indices], -100.0, 100.0)
        return self.action * self.cfg.action_scale + self.cfg.default_joints

    def run(self):
        pre_tic = 0
        gait_process = 0
        duration_second = self.cfg.decimation * self.cfg.dt
        duration_millisecond = duration_second * 1000
        timer = NanoSleep(duration_millisecond)
        pbar = tqdm(range(int(0xFFFF_FFF0 / duration_second)), desc="E1 running...")
        start = time.perf_counter()
        for _ in pbar:
            start_time = time.perf_counter()
            self.get_leg_state()
            if self.rc.state.LT > 64:
                print("Emergency stop")
                exit()
            q, dq, obs = self.get_obs(gait_process)
            self.hist_obs.append(obs)
            self.target_q = self.get_action(self.hist_obs.get())
            self.target_q = np.append(self.target_q, 0.0)
            for idx in range(self.legActions):
                self.legCommand.position[idx] = self.target_q[idx]
            self.set_leg_command()
            pbar.set_postfix(
                realCycle=f"{self.legState.system_tic - pre_tic}ms",
                calculateTime=f"{(time.perf_counter() - start_time) * 1000:.3f}ms",
                runTime=f"{(time.perf_counter() - start):.3f}s",
            )
            pre_tic = self.legState.system_tic
            gait_process = np.fmod(gait_process + duration_second * self.gait_frequency, 1.0)
            timer.waiting(start_time)
        self.set_leg_path(1, self.cfg.default_joints)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gamepad", action="store_true")
    args = parser.parse_args()

    mybot = Sim2Real(use_gamepad=not args.no_gamepad)
    mybot.init_robot()
    time.sleep(1)
    mybot.run()
