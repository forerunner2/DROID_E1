import argparse
import math
import sys
import threading
import time

import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from base.LegBase import LegBase
from base.Base import get_command
from base.Base import NanoSleep, euler_to_quaternion, quat_rotate_inverse
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
    "waist_yaw_joint",
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
    "waist_yaw_joint",
]
Isaac_to_Real_indices = [IsaacLabJointOrder.index(joint) for joint in RealJointOrder]
Real_to_Isaac_indices = [RealJointOrder.index(joint) for joint in IsaacLabJointOrder]


class CommandInput:
    def __init__(self, default_cmd, default_gait):
        self._lock = threading.Lock()
        self._command = list(default_cmd)
        self._gait_frequency = float(default_gait)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        print("CLI input: vx vy yaw | 'stop' | 'freq <hz>' | 'exit'")
        while self._running:
            line = sys.stdin.readline()
            if not line:
                time.sleep(0.1)
                continue
            line = line.strip()
            if not line:
                continue
            lower = line.lower()
            if lower in ("exit", "quit"):
                self._running = False
                break
            if lower == "stop":
                with self._lock:
                    self._command = [0.0, 0.0, 0.0]
                continue
            if lower.startswith("freq"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        freq = float(parts[1])
                        with self._lock:
                            self._gait_frequency = freq
                        print(f"gait_frequency = {freq}")
                    except ValueError:
                        print("Invalid freq. Example: freq 1.5")
                else:
                    print("Usage: freq <value>")
                continue
            for sep in (",", ";"):
                line = line.replace(sep, " ")
            parts = line.split()
            if len(parts) != 3:
                print("Input format: vx vy yaw (e.g. 0.5 0 0)")
                continue
            try:
                vx, vy, yaw = map(float, parts)
            except ValueError:
                print("Invalid numbers. Example: 0.5 0 0")
                continue
            with self._lock:
                self._command = [vx, vy, yaw]

    def get(self):
        with self._lock:
            return list(self._command), float(self._gait_frequency)

    def running(self):
        return self._running


class Sim2Real(LegBase):
    def __init__(self, default_gait: float, max_increment: float):
        LegBase.__init__(self)
        self.num_actions = len(RealJointOrder)
        self.num_observations = 11 + 3 * self.num_actions
        self.gait_frequency = float(default_gait)
        self.cfg = load_configuration("policies/env_cfg.json", RealJointOrder)
        self.run_flag = True
        self.max_increment = float(max_increment)

        self.command = [0.0, 0.0, 0.0]
        self.target_q = np.zeros(self.num_actions, dtype=np.double)
        self.action = np.zeros(self.num_actions, dtype=np.double)
        self.onnx_policy = ort.InferenceSession(onnx_mode_path)
        self.hist_obs = CircularBuffer(self.num_observations, self.cfg.hist_length)
        self.cli = CommandInput(self.command, default_gait)

    def init_robot(self):
        print("default_joints:", self.cfg.default_joints)
        init_pos = np.array(self.cfg.default_joints, dtype=np.float32)
        if init_pos.shape[0] < self.num_actions:
            pad = np.zeros(self.num_actions - init_pos.shape[0], dtype=np.float32)
            init_pos = np.concatenate([init_pos, pad])
        self.set_leg_path(1, init_pos)
        print("CLI control ready. Type commands in this terminal.")

    def update_rc_command(self):
        if not self.cli.running():
            self.run_flag = False
            return
        target_cmd, gait_freq = self.cli.get()
        self.command[0] = get_command(self.command[0], target_cmd[0], self.max_increment)
        self.command[1] = get_command(self.command[1], target_cmd[1], self.max_increment)
        self.command[2] = get_command(self.command[2], target_cmd[2], self.max_increment)
        self.gait_frequency = gait_freq

    def get_gravity_orientation_from_rpy(self, roll, pitch):
        rot = R.from_euler("xy", [roll, pitch])
        g_world = np.array([0, 0, -1])
        g_local = rot.inv().apply(g_world)
        return g_local

    def get_obs(self, gait_process):
        q = np.array(self.legState.position[: self.num_actions])
        dq = np.array(self.legState.velocity[: self.num_actions])

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
        offset = 11
        pos_end = offset + self.num_actions
        vel_end = pos_end + self.num_actions
        act_end = vel_end + self.num_actions
        obs[offset:pos_end] = (q - self.cfg.default_joints)[Real_to_Isaac_indices]
        obs[pos_end:vel_end] = dq[Real_to_Isaac_indices]
        obs[vel_end:act_end] = self.action[Real_to_Isaac_indices]
        obs = np.clip(obs, -100, 100)
        return q, dq, obs

    def get_action(self, obs):
        obs = [np.array(obs, dtype=np.float32)]
        action = np.array(self.onnx_policy.run(None, {"obs": obs})[0].tolist()[0])
        self.action = np.clip(action[Isaac_to_Real_indices], -100.0, 100.0)
        return self.action * self.cfg.action_scale + self.cfg.default_joints

    def run(self, show_progress: bool):
        pre_tic = 0
        gait_process = 0
        duration_second = self.cfg.decimation * self.cfg.dt
        duration_millisecond = duration_second * 1000
        timer = NanoSleep(duration_millisecond)
        pbar = tqdm(
            range(int(0xFFFF_FFF0 / duration_second)),
            desc="E1 running...",
            disable=not show_progress,
        )
        start = time.perf_counter()
        for _ in pbar:
            if not self.run_flag:
                break
            start_time = time.perf_counter()
            self.get_leg_state()
            q, dq, obs = self.get_obs(gait_process)
            self.hist_obs.append(obs)
            self.target_q = self.get_action(self.hist_obs.get())
            for idx in range(self.legActions):
                self.legCommand.position[idx] = self.target_q[idx]
            self.set_leg_command()
            if show_progress:
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
    parser.add_argument("--gait-frequency", type=float, default=1.5)
    parser.add_argument("--max-increment", type=float, default=0.05)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    mybot = Sim2Real(default_gait=args.gait_frequency, max_increment=args.max_increment)
    mybot.init_robot()
    time.sleep(1)
    mybot.run(show_progress=not args.no_progress)
