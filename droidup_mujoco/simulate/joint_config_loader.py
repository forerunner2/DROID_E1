from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

_LOGGER = logging.getLogger(__name__)


@dataclass
class JointConfig:
    dof_stiffness: np.ndarray
    dof_damping: np.ndarray
    effort_limit: np.ndarray
    default_joints: np.ndarray
    standing_height: float
    action_scale: float
    dt: float
    hist_length: int
    decimation: int


def _resolve_value(
    joint_meta: Mapping[str, float],
    defaults: Mapping[str, float],
    key: str,
    *,
    fallback: float = 0.0,
) -> float:
    if key in joint_meta:
        return float(joint_meta[key])
    if key in defaults:
        return float(defaults[key])
    return fallback


def load_joint_config(path: str | Path, joint_order: Sequence[str]) -> JointConfig:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    defaults = raw.get("defaults", {})
    joints = raw.get("joints", {})
    count = len(joint_order)

    stiffness = np.zeros(count, dtype=np.float32)
    damping = np.zeros(count, dtype=np.float32)
    effort = np.zeros(count, dtype=np.float32)
    default_pos = np.zeros(count, dtype=np.float32)

    for idx, name in enumerate(joint_order):
        meta = joints.get(name, {})
        stiffness[idx] = _resolve_value(meta, defaults, "kp")
        damping[idx] = _resolve_value(meta, defaults, "kd")
        effort[idx] = _resolve_value(meta, defaults, "effort")
        if "default" in meta:
            default_pos[idx] = float(meta["default"])
        elif "default_deg" in meta:
            default_pos[idx] = float(np.deg2rad(meta["default_deg"]))
        else:
            default_pos[idx] = _resolve_value(meta, defaults, "default")
        if not meta:
            _LOGGER.debug(
                "关节 %s 未在 %s 中配置，使用默认值 kp=%.2f kd=%.2f effort=%.2f default=%.3f",
                name,
                cfg_path,
                stiffness[idx],
                damping[idx],
                effort[idx],
                default_pos[idx],
            )

    return JointConfig(
        dof_stiffness=stiffness,
        dof_damping=damping,
        effort_limit=effort,
        default_joints=default_pos,
        standing_height=float(raw.get("standing_height", 0.9)),
        action_scale=float(raw.get("action_scale", 1.0)),
        dt=float(raw.get("dt", 0.001)),
        hist_length=int(raw.get("hist_length", 1)),
        decimation=int(raw.get("decimation", 1)),
    )

