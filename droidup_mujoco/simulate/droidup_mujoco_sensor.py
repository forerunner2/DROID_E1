from __future__ import annotations

import argparse
import numpy as np

from MujoBase import MujoBase
import droidup_mujoco


def _get_robot_state_sensor(self: MujoBase):
    q = self.data.qpos.astype(np.float32)[7:]
    dq = self.data.qvel.astype(np.float32)[6:]

    ang_vel = None
    quat = None
    try:
        ang_vel = self.data.sensor("gyro").data.astype(np.float32)
    except Exception:
        ang_vel = None
    try:
        quat_raw = self.data.sensor("bq").data.astype(np.float32)
        quat = quat_raw[[1, 2, 3, 0]]
    except Exception:
        quat = None

    if ang_vel is None:
        ang_vel = self.data.qvel[3:6].astype(np.float32)

    if quat is None:
        quat = self.data.qpos[3:7].astype(np.float32)
        quat[:] = quat[[1, 2, 3, 0]]

    return q, dq, quat, ang_vel


def main(argv: list[str] | None = None) -> None:
    # Monkey-patch MujoBase.get_robot_state to use sensors when available
    MujoBase.get_robot_state = _get_robot_state_sensor  # type: ignore[assignment]

    parser = droidup_mujoco.build_arg_parser()
    args = parser.parse_args(argv)
    droidup_mujoco.main(argv)


if __name__ == "__main__":
    main()
