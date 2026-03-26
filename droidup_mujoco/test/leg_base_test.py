# Copyright (c) 2025 DroidUp, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the DroidUp Software
# Development Kit License (20250922-DROIDUPSDK-SL).

import math
import argparse
import numpy as np
from droidup.client.legbase import LegBase

def _to_list(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return value

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="DroidUp LegBase Test")
    parser.add_argument("--host", default="localhost", help="Server Domain Name Or IP")
    parser.add_argument("--port", default=50051, type=int, help="Server Port")
    args = parser.parse_args()

    print("\n===== DroidUp LegBase Test =====")
    Leg = LegBase(
        _host=args.host,
        _port=args.port
    )
    period = 1  # 总时间
    kpq = np.zeros(Leg.num_actions)
    kdq = np.zeros(Leg.num_actions)
    dt0 = np.zeros(Leg.num_actions)
    dt1 = np.zeros(Leg.num_actions)
    dt2 = np.zeros(Leg.num_actions)
    config = Leg.get_joint_config()
    # 填充 dt1 和 dt2 列表
    if Leg.num_actions == 10:
        kpq = [20, 100, 100, 100, 20, 20, 100, 100, 100, 20]
        kdq = [1, 4, 4, 4, 2, 1, 4, 4, 4, 2]
        dt1 = [round(math.radians(d), 4) for d in [0, 0, 30, -60, 30, 0, 0, 0, 0, 0]]
        dt2 = [round(math.radians(d), 4) for d in [0, 0, 0, 0, 0, 0, 0, 30, -60, 30]]
    elif Leg.num_actions == 12:
        kpq = [20, 100, 100, 100, 20, 20, 20, 100, 100, 100, 20, 20]
        kdq = [1, 4, 4, 4, 2, 2, 1, 4, 4, 4, 2, 2]
        dt1 = [round(math.radians(d), 4) for d in [0, 0, 30, -60, 30, -0.3, 0, 0, 0, 0, 0, 0.3]]
        dt2 = [round(math.radians(d), 4) for d in [0, 0, 0, 0, 0, 0.3, 0, 0, 30, -60, 30, -0.3]]
    elif Leg.num_actions == 13:
        kpq = config.kp
        kdq = config.kd
        dt1 = [round(math.radians(d), 4) for d in [0, 0, 30, -60, 30, -0.3, 0, 0, 0, 0, 0, 0.3]] + [0.0]
        dt2 = [round(math.radians(d), 4) for d in [0, 0, 0, 0, 0, 0.3, 0, 0, 30, -60, 30, -0.3]] + [0.0]
    elif Leg.num_actions == 14:
        kpq = config.kp
        kdq = config.kd
        dt1 = [round(math.radians(d), 4) for d in [-30, 0, 0, 60, 60, 10, -30, 0, 0, 0, 0, 0, 0, 0]]
        dt2 = [round(math.radians(d), 4) for d in [0, 0, 0, 0, 0, 0, 0, -30, 0, 0, 60, 60, 10, -30]]
    kpq = _to_list(kpq)
    kdq = _to_list(kdq)
    dt0 = _to_list(dt0)
    dt1 = _to_list(dt1)
    dt2 = _to_list(dt2)
    Leg.set_joint_parameters(kpq, kdq, _to_list(config.imax))
    # 执行关节规划
    for i in range(1000):
        print("wave round %d" % (i * 2 + 1))
        Leg.set_joint_path(period, dt1)
        print("wave round %d" % (i * 2 + 2))
        Leg.set_joint_path(period, dt2)
    print("return to zero")
    Leg.set_joint_path(period, dt0)
    print("Test complete ...")
    Leg.disconnect()


if __name__ == "__main__":
    main()
