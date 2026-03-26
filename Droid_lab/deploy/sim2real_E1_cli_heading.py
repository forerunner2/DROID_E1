import argparse
import json
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


ENV_CFG_PATH = "policies/env_cfg.json"
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


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def load_heading_cfg(env_cfg_path: str):
    try:
        with open(env_cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        commands = data.get("commands", {})
        heading_cmd = bool(commands.get("heading_command", False))
        heading_k = float(commands.get("heading_control_stiffness", 0.0))
        ang_range = commands.get("ranges", {}).get("ang_vel_z", [-1.0, 1.0])
        if isinstance(ang_range, list) and len(ang_range) == 2:
            ang_min = float(ang_range[0])
            ang_max = float(ang_range[1])
        else:
            ang_min, ang_max = -1.0, 1.0
        return heading_cmd, heading_k, (ang_min, ang_max)
    except Exception as exc:
        print(f"[WARN] Failed to read heading config from {env_cfg_path}: {exc}")
        return False, 0.0, (-1.0, 1.0)


class CommandInput:
    def __init__(self, default_cmd, default_gait):
        self._lock = threading.Lock()
        self._command = list(default_cmd)
        self._gait_frequency = float(default_gait)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        print("CLI input: vx vy heading(rad) | 'stop' | 'freq <hz>' | 'exit'")
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
                print("Input format: vx vy heading(rad) (e.g. 0.5 0 0)")
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
        self.num_actions = 12
        self.num_observations = 47
        self.gait_frequency = float(default_gait)
        self.cfg = load_configuration(ENV_CFG_PATH, RealJointOrder)
        self.run_flag = True
        self.max_increment = float(max_increment)

        self.command = [0.0, 0.0, 0.0]
        self.target_q = np.zeros(self.num_actions, dtype=np.double)
        self.action = np.zeros(self.num_actions, dtype=np.double)
        self.onnx_policy = ort.InferenceSession(onnx_mode_path)
        self.hist_obs = CircularBuffer(self.num_observations, self.cfg.hist_length)
        self.cli = CommandInput(self.command, default_gait)
        self.heading_command, self.heading_k, self.ang_vel_range = load_heading_cfg(ENV_CFG_PATH)
        print(
            f"[INFO] heading_command={self.heading_command}, "
            f"heading_control_stiffness={self.heading_k}, "
            f"ang_vel_z_range={self.ang_vel_range}"
        )

    def init_robot(self):
        print("default_joints:", self.cfg.default_joints)
        init_pos = np.append(self.cfg.default_joints, 0.0)
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
        q = np.array(self.legState.position[:12])
        dq = np.array(self.legState.velocity[:12])

        base_euler = np.array(self.legState.imu_euler)
        base_ang_vel = np.array(self.legState.imu_gyro)

        base_euler[base_euler > math.pi] -= 2 * math.pi
        eq = euler_to_quaternion(base_euler[0], base_euler[1], base_euler[2])
        eq = np.array(eq, dtype=np.double)
        project_gravity = quat_rotate_inverse(eq, np.array([0.0, 0.0, -1]))

        self.update_rc_command()

        command = list(self.command)
        if self.heading_command:
            current_yaw = float(base_euler[2])
            heading_error = wrap_to_pi(command[2] - current_yaw)
            yaw_rate = self.heading_k * heading_error
            yaw_rate = float(np.clip(yaw_rate, self.ang_vel_range[0], self.ang_vel_range[1]))
            command[2] = yaw_rate

        obs = np.zeros([self.num_observations], dtype=np.float32)
        obs[0:3] = base_ang_vel
        obs[3:6] = project_gravity
        obs[6:9] = command
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
            self.target_q = np.append(self.target_q, 0.0)
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
