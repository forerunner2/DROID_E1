from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_E1_JOINTS = [
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


def _set_matching(values: Dict[str, Optional[float]], pattern: re.Pattern, value: Optional[float]) -> None:
    for key in values:
        if pattern.match(key):
            values[key] = 0.0 if value is None else float(value)


def _first_non_none(values: Iterable[Optional[float]], fallback: float = 0.0) -> float:
    for v in values:
        if v is not None:
            return float(v)
    return float(fallback)


def _most_common(values: Iterable[Optional[float]], fallback: float = 0.0) -> float:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return float(fallback)
    counts = Counter(filtered)
    return float(counts.most_common(1)[0][0])


def _load_env_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_joint_order(args: argparse.Namespace) -> List[str]:
    if args.joint_order:
        return [item.strip() for item in args.joint_order.split(",") if item.strip()]
    if args.joint_order_file:
        content = Path(args.joint_order_file).read_text(encoding="utf-8").strip()
        if not content:
            return []
        if content.startswith("["):
            return json.loads(content)
        return [line.strip() for line in content.splitlines() if line.strip()]
    return list(DEFAULT_E1_JOINTS)


def _build_joint_maps(env_cfg: dict, joint_order: List[str]) -> dict:
    joint_kp: Dict[str, Optional[float]] = {name: None for name in joint_order}
    joint_kd: Dict[str, Optional[float]] = {name: None for name in joint_order}
    joint_effort: Dict[str, Optional[float]] = {name: None for name in joint_order}
    joint_default: Dict[str, Optional[float]] = {name: None for name in joint_order}

    actuators = env_cfg["scene"]["robot"]["actuators"]
    for group_cfg in actuators.values():
        exprs = group_cfg["joint_names_expr"]
        for expr in exprs:
            regex = re.compile(expr)
            _set_matching(joint_kp, regex, group_cfg["stiffness"][expr])
            _set_matching(joint_kd, regex, group_cfg["damping"][expr])
            _set_matching(joint_effort, regex, group_cfg["effort_limit_sim"][expr])

    default_joint_data = env_cfg["scene"]["robot"]["init_state"]["joint_pos"]
    for expr, value in default_joint_data.items():
        regex = re.compile(expr)
        _set_matching(joint_default, regex, value)

    return {
        "kp": joint_kp,
        "kd": joint_kd,
        "effort": joint_effort,
        "default": joint_default,
    }


def _build_output(env_cfg: dict, joint_order: List[str], *, defaults_mode: str) -> dict:
    maps = _build_joint_maps(env_cfg, joint_order)

    if defaults_mode == "mode":
        default_kp = _most_common(maps["kp"].values())
        default_kd = _most_common(maps["kd"].values())
        default_effort = _most_common(maps["effort"].values())
        default_pos = _most_common(maps["default"].values())
    else:
        default_kp = _first_non_none(maps["kp"].values())
        default_kd = _first_non_none(maps["kd"].values())
        default_effort = _first_non_none(maps["effort"].values())
        default_pos = _first_non_none(maps["default"].values())

    joints_out: Dict[str, dict] = {}
    missing = []
    for name in joint_order:
        kp = maps["kp"][name]
        kd = maps["kd"][name]
        effort = maps["effort"][name]
        default = maps["default"][name]
        if kp is None or kd is None or effort is None or default is None:
            missing.append(name)
        joints_out[name] = {
            "kp": float(default_kp if kp is None else kp),
            "kd": float(default_kd if kd is None else kd),
            "effort": float(default_effort if effort is None else effort),
            "default": float(default_pos if default is None else default),
        }

    out = {
        "dt": float(env_cfg["sim"]["dt"]),
        "decimation": int(env_cfg["sim"]["decimation"]),
        "standing_height": float(env_cfg["scene"]["robot"]["init_state"]["pos"][2]),
        "action_scale": float(env_cfg["robot"]["action_scale"]),
        "hist_length": int(env_cfg["robot"]["actor_obs_history_length"]),
        "defaults": {
            "kp": float(default_kp),
            "kd": float(default_kd),
            "effort": float(default_effort),
            "default": float(default_pos),
        },
        "joints": joints_out,
    }

    return out, missing


def _find_workspace_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "Droid_lab").exists() and (parent / "droidup_mujoco").exists():
            return parent
    return start


def build_arg_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    default_env = script_dir.parent / "policies" / "env_cfg.json"

    workspace_root = _find_workspace_root(script_dir)
    default_out = workspace_root / "droidup_mujoco" / "simulate" / "configs" / "e1_from_env_cfg.json"

    parser = argparse.ArgumentParser(description="Convert IsaacLab env_cfg.json to droidup_mujoco joint config JSON")
    parser.add_argument("--env-cfg", default=str(default_env), help="Path to env_cfg.json")
    parser.add_argument("--output", default=str(default_out), help="Output joint config JSON path")
    parser.add_argument("--joint-order", default="", help="Comma-separated joint order list")
    parser.add_argument("--joint-order-file", default="", help="File with joint names (one per line or JSON list)")
    parser.add_argument("--defaults-mode", choices=["first", "mode"], default="first",
                        help="How to derive defaults if a joint is missing values")
    parser.add_argument("--strict", action="store_true", help="Fail if any joint is missing values")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    env_path = Path(args.env_cfg).resolve()
    out_path = Path(args.output).resolve()

    joint_order = _parse_joint_order(args)
    if not joint_order:
        raise SystemExit("Joint order is empty. Provide --joint-order or --joint-order-file.")

    env_cfg = _load_env_cfg(env_path)
    output, missing = _build_output(env_cfg, joint_order, defaults_mode=args.defaults_mode)

    if missing:
        message = "Missing values for joints: " + ", ".join(missing)
        if args.strict:
            raise SystemExit(message)
        print("[WARN] " + message)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("[OK] Written:", out_path)
    print("[INFO] dt=", output["dt"], "decimation=", output["decimation"], "hist_length=", output["hist_length"])


if __name__ == "__main__":
    main()
