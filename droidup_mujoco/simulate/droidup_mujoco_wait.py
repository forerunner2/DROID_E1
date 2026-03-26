from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import grpc
import numpy as np

from MujoBase import MujoBase
from Base import quaternion_to_euler_array
from Config import Config
from droidup.api import arm_service_pb2_grpc as arm_pb2_grpc
from droidup.api import droidup_msg_pb2 as msg_pb2
from droidup.api import leg_service_pb2_grpc as leg_pb2_grpc
from joint_config_loader import load_joint_config


_LOGGER = logging.getLogger("mujoco.grpc")


@dataclass
class JointGroup:
    name: str
    joint_names: Sequence[str]
    actuator_indices: Sequence[int]

    @property
    def size(self) -> int:
        return len(self.actuator_indices)


class VideoRecorder:
    def __init__(
        self,
        model,
        data,
        path: str,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        try:
            import imageio.v2 as imageio
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "Video recording requires imageio. Install with: pip install imageio[ffmpeg]"
            ) from exc

        import mujoco

        self._data = data
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._writer = imageio.get_writer(path, fps=fps)
        self._frame_interval = 1.0 / float(fps)
        self._next_frame_time = 0.0

    def maybe_capture(self, sim_time_s: float) -> None:
        if sim_time_s + 1e-9 < self._next_frame_time:
            return
        try:
            self._renderer.cam.lookat[:] = self._data.qpos.astype(np.float32)[0:3]
        except Exception:
            pass
        self._renderer.update_scene(self._data)
        frame = self._renderer.render()
        self._writer.append_data(frame)
        self._next_frame_time += self._frame_interval

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def _parse_joint_names(raw: str) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


class RobotSimulation:
    def __init__(
        self,
        model_path: str,
        joint_cfg_path: str,
        leg_joint_names: Sequence[str],
        arm_joint_names: Sequence[str],
        headless: bool,
        viewer_size: int,
        record: bool,
        record_path: str,
        record_width: int,
        record_height: int,
        record_fps: int,
        wait_for_command: bool,
    ) -> None:
        self._mujoco = MujoBase(model_path, headless=headless)
        self._actuator_names = self._mujoco.get_joint_names()
        self._name_to_index: Dict[str, int] = {
            name: idx for idx, name in enumerate(self._actuator_names)
        }
        self.policy_cfg = load_joint_config(joint_cfg_path, self._actuator_names)
        # Override MuJoCo PD parameters with policy configuration
        self._mujoco.model.opt.timestep = float(self.policy_cfg.dt)
        self._mujoco.cfg.default_joints = self.policy_cfg.default_joints.astype(np.float32)
        self._mujoco.cfg.dof_stiffness = self.policy_cfg.dof_stiffness.astype(np.float32)
        self._mujoco.cfg.dof_damping = self.policy_cfg.dof_damping.astype(np.float32)
        self._mujoco.cfg.effort_limit = self.policy_cfg.effort_limit.astype(np.float32)

        self._leg_group = self._build_group(
            "legs", leg_joint_names, Config.num_leg_actions, offset=0
        )
        self._arm_group = self._build_group(
            "arms",
            arm_joint_names,
            Config.num_arm_actions,
            offset=self._leg_group.size,
        )

        self._target_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._desired_q = self.policy_cfg.default_joints.astype(np.float32).copy()
        self._sim_time_ms = 0.0

        self._leg_config_msg = self._build_config_message(self._leg_group)
        self._arm_config_msg = self._build_config_message(self._arm_group)
        self._leg_state_cache = msg_pb2.DroidStateResponse()
        self._arm_state_cache = msg_pb2.DroidArmResponse()

        self._running = threading.Event()
        self._physics_thread: threading.Thread | None = None
        self._sim_dt = float(self.policy_cfg.dt)
        self._recorder: VideoRecorder | None = None
        self._wait_for_command = wait_for_command
        self._command_received = threading.Event()

        if record:
            record_dir = os.path.dirname(record_path)
            if record_dir:
                os.makedirs(record_dir, exist_ok=True)
            self._recorder = VideoRecorder(
                self._mujoco.model,
                self._mujoco.data,
                path=record_path,
                width=record_width,
                height=record_height,
                fps=record_fps,
            )
            _LOGGER.info("Recording enabled: %s", record_path)

        _LOGGER.info(
            "Loaded MuJoCo model with %d actuators", len(self._actuator_names)
        )
        _LOGGER.info(
            "Leg joints (%d): %s", self._leg_group.size, list(self._leg_group.joint_names)
        )
        _LOGGER.info(
            "Arm joints (%d): %s", self._arm_group.size, list(self._arm_group.joint_names)
        )
        if self._wait_for_command:
            _LOGGER.info("Waiting for first control command before stepping physics.")

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._physics_thread = threading.Thread(
            target=self._physics_loop, name="mujoco-physics", daemon=True
        )
        self._physics_thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._physics_thread and self._physics_thread.is_alive():
            self._physics_thread.join(timeout=2.0)
        if self._recorder is not None:
            self._recorder.close()
            self._recorder = None

    def apply_leg_command(self, request: msg_pb2.DroidCommandRequest) -> None:
        self._apply_group_command(self._leg_group, request)

    def apply_arm_command(self, request: msg_pb2.DroidCommandRequest) -> None:
        self._apply_group_command(self._arm_group, request)

    def get_leg_config(self) -> msg_pb2.DroidConfigs:
        cfg = msg_pb2.DroidConfigs()
        cfg.CopyFrom(self._leg_config_msg)
        return cfg

    def get_arm_config(self) -> msg_pb2.DroidConfigs:
        cfg = msg_pb2.DroidConfigs()
        cfg.CopyFrom(self._arm_config_msg)
        return cfg

    def get_leg_state(self) -> msg_pb2.DroidStateResponse:
        state = msg_pb2.DroidStateResponse()
        with self._state_lock:
            state.CopyFrom(self._leg_state_cache)
        return state

    def get_arm_state(self) -> msg_pb2.DroidArmResponse:
        state = msg_pb2.DroidArmResponse()
        with self._state_lock:
            state.CopyFrom(self._arm_state_cache)
        return state

    def _build_group(
        self,
        name: str,
        joint_names: Sequence[str],
        default_count: int,
        *,
        offset: int = 0,
    ) -> JointGroup:
        if not joint_names:
            joint_names = self._actuator_names[offset : offset + default_count]
        indices: List[int] = []
        for joint in joint_names:
            if joint not in self._name_to_index:
                raise ValueError(f"Joint {joint} not found in MuJoCo actuators")
            indices.append(self._name_to_index[joint])
        return JointGroup(name=name, joint_names=list(joint_names), actuator_indices=indices)

    def _build_config_message(self, group: JointGroup) -> msg_pb2.DroidConfigs:
        cfg = msg_pb2.DroidConfigs()
        for joint, idx in zip(group.joint_names, group.actuator_indices):
            cfg.joint_name.append(joint)
            cfg.pzero.append(float(self.policy_cfg.default_joints[idx]))
            cfg.pmin.append(-np.pi)
            cfg.pmax.append(np.pi)
            cfg.imax.append(float(self.policy_cfg.effort_limit[idx]))
            cfg.kp.append(float(self.policy_cfg.dof_stiffness[idx]))
            cfg.kd.append(float(self.policy_cfg.dof_damping[idx]))
        return cfg

    def _apply_group_command(
        self, group: JointGroup, request: msg_pb2.DroidCommandRequest
    ) -> None:
        if group.size == 0:
            return
        target_vector = np.array(request.position[: group.size], dtype=np.float32)
        if target_vector.size < group.size:
            padded = np.zeros(group.size, dtype=np.float32)
            padded[: target_vector.size] = target_vector
            target_vector = padded
        with self._target_lock:
            for local_idx, actuator_idx in enumerate(group.actuator_indices):
                self._desired_q[actuator_idx] = target_vector[local_idx]
        self._command_received.set()

    def _physics_loop(self) -> None:
        next_wakeup = time.perf_counter()
        while self._running.is_set():
            if self._wait_for_command and not self._command_received.is_set():
                q, dq, quat, ang_vel = self._mujoco.get_robot_state()
                self._update_state_cache(q, dq, quat, ang_vel)
                time.sleep(0.001)
                next_wakeup = time.perf_counter()
                continue

            q, dq, quat, ang_vel = self._mujoco.get_robot_state()
            with self._target_lock:
                targets = self._desired_q.copy()
            self._mujoco.set_robot_state(targets, q, dq)
            self._sim_time_ms += self._sim_dt * 1000.0
            self._update_state_cache(q, dq, quat, ang_vel)
            if self._recorder is not None:
                self._recorder.maybe_capture(self._sim_time_ms / 1000.0)
            next_wakeup += self._sim_dt
            sleep_duration = next_wakeup - time.perf_counter()
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            else:
                next_wakeup = time.perf_counter()

    def _update_state_cache(
        self, q: np.ndarray, dq: np.ndarray, quat: np.ndarray, ang_vel: np.ndarray
    ) -> None:
        leg_state = msg_pb2.DroidStateResponse()
        leg_state.system_tic = int(self._sim_time_ms)
        torque = self._mujoco.data.ctrl.copy()
        for idx in range(self._leg_group.size):
            actuator_idx = self._leg_group.actuator_indices[idx]
            leg_state.position.append(float(q[actuator_idx]))
            leg_state.velocity.append(float(dq[actuator_idx]))
            leg_state.torque.append(float(torque[actuator_idx]))
            leg_state.temperature.append(25.0)
            leg_state.abs_encoder.append(float(q[actuator_idx]))
            leg_state.pack_loss.append(0)
        leg_state.foot_force.extend([0.0, 0.0, 0.0, 0.0])
        leg_state.imu_stamp = int(self._sim_time_ms)
        leg_state.imu_acc.extend([0.0, 0.0, -9.81])
        leg_state.imu_euler.extend(quaternion_to_euler_array(quat).tolist())
        leg_state.imu_gyro.extend(ang_vel.tolist())
        leg_state.bus_voltage = 50.0
        leg_state.bus_current = 2.5
        leg_state.bus_energy = 0.0

        arm_state = msg_pb2.DroidArmResponse()
        for idx in range(self._arm_group.size):
            actuator_idx = self._arm_group.actuator_indices[idx]
            arm_state.position.append(float(q[actuator_idx]))
            arm_state.velocity.append(float(dq[actuator_idx]))
            arm_state.torque.append(float(torque[actuator_idx]))
        # arm_state.finger.extend([0.0] * 12)

        with self._state_lock:
            self._leg_state_cache.CopyFrom(leg_state)
            self._arm_state_cache.CopyFrom(arm_state)


class LegService(leg_pb2_grpc.LegServiceServicer):
    def __init__(self, sim: RobotSimulation) -> None:
        self._sim = sim

    async def GetLegConfig(self, request, context):  # noqa: N802
        return self._sim.get_leg_config()

    async def GetLegState(self, request, context):  # noqa: N802
        return self._sim.get_leg_state()

    async def SetLegCommand(self, request, context):  # noqa: N802
        self._sim.apply_leg_command(request)
        return msg_pb2.Empty()


class ArmService(arm_pb2_grpc.ArmServiceServicer):
    def __init__(self, sim: RobotSimulation) -> None:
        self._sim = sim

    async def GetArmConfig(self, request, context):  # noqa: N802
        return self._sim.get_arm_config()

    async def GetArmState(self, request, context):  # noqa: N802
        return self._sim.get_arm_state()

    async def SetArmCommand(self, request, context):  # noqa: N802
        self._sim.apply_arm_command(request)
        return msg_pb2.Empty()


async def _serve_async(args: argparse.Namespace) -> None:
    sim = RobotSimulation(
        model_path=args.model,
        joint_cfg_path=args.joint_cfg,
        leg_joint_names=_parse_joint_names(args.leg_joints),
        arm_joint_names=_parse_joint_names(args.arm_joints),
        headless=not args.viewer,
        viewer_size=args.viewer_size,
        record=args.record,
        record_path=args.record_path,
        record_width=args.record_width,
        record_height=args.record_height,
        record_fps=args.record_fps,
        wait_for_command=args.wait_for_command,
    )
    sim.start()

    server = grpc.aio.server(options=[("grpc.max_send_message_length", -1), ("grpc.max_receive_message_length", -1)])
    leg_pb2_grpc.add_LegServiceServicer_to_server(LegService(sim), server)
    if sim._arm_group.size > 0 and args.enable_arm_service:  # pylint: disable=protected-access
        arm_pb2_grpc.add_ArmServiceServicer_to_server(ArmService(sim), server)
    server.add_insecure_port(f"{args.host}:{args.leg_port}")
    if args.arm_port != args.leg_port:
        server.add_insecure_port(f"{args.host}:{args.arm_port}")

    await server.start()
    _LOGGER.info(
        "gRPC server started on %s:%d (legs) and %s:%d (arms)",
        args.host,
        args.leg_port,
        args.host,
        args.arm_port,
    )

    stop_event = asyncio.Event()

    def _handle_signal(*_):  # noqa: ANN001
        _LOGGER.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    await stop_event.wait()
    await server.stop(grace=2.0)
    sim.stop()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MuJoCo gRPC bridge")
    parser.add_argument("--model", default="assets/droidup/e1/scene.xml", help="Path to MuJoCo XML/scene file")
    default_joint_cfg = os.path.join(os.path.dirname(__file__), "configs/e1_locomotion.json")
    parser.add_argument("--joint-cfg", default=default_joint_cfg, help="Path to simplified joint config JSON")
    parser.add_argument("--host", default="0.0.0.0", help="Host/IP to bind gRPC services")
    parser.add_argument("--leg-port", type=int, default=50051, help="Port for leg service")
    parser.add_argument("--arm-port", type=int, default=50052, help="Port for arm service")
    parser.add_argument("--leg-joints", default="", help="Comma separated joint names for leg service order")
    parser.add_argument("--arm-joints", default="", help="Comma separated joint names for arm service order")
    parser.add_argument("--viewer", default=True, action="store_true", help="Show MuJoCo viewer window (on by default)")
    parser.add_argument("--viewer-size", type=int, default=1200, help="Viewer window resolution when viewer is enabled")
    parser.add_argument("--enable-arm-service", action="store_true", help="Expose arm gRPC service if arm joints exist")
    parser.add_argument("--record", action="store_true", help="Record MuJoCo video to a file")
    parser.add_argument("--record-path", default="mujoco_recording.mp4", help="Output video path")
    parser.add_argument("--record-fps", type=int, default=30, help="Recording frames per second")
    parser.add_argument("--record-width", type=int, default=1280, help="Recording width")
    parser.add_argument("--record-height", type=int, default=720, help="Recording height")
    parser.add_argument(
        "--wait-for-command",
        action="store_true",
        help="Wait for the first control command before stepping physics",
    )
    parser.set_defaults(enable_arm_service=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        asyncio.run(_serve_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
