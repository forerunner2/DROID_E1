# Copyright (c) 2025 DroidUp, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the DroidUp Software
# Development Kit License (20250922-DROIDUPSDK-SL).

import math
import argparse
import numpy as np
from droidup.client.armbase import ArmBase

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="DroidUp ArmBase Test")
    parser.add_argument("--host", default="localhost", help="Server Domain Name Or IP")
    parser.add_argument("--port", default=50052, type=int, help="Server Port")
    args = parser.parse_args()

    print("\n===== DroidUp ArmBase Test =====")
    Arm = ArmBase(
        _host=args.host,
        _port=args.port
    )
    config = Arm.get_joint_config()
    Arm.set_joint_parameters(config.kp, config.kd, config.imax)
    period = 1  # 总时间
    dt0 = np.zeros(Arm.num_actions)
    dt1 = np.zeros(Arm.num_actions)
    dt2 = np.zeros(Arm.num_actions)
    if Arm.num_actions == 2:
        dt0 = [round(math.radians(d), 4) for d in [-30, -30]]
        dt1 = [round(math.radians(d), 4) for d in [-30, 30]]
        dt2 = [round(math.radians(d), 4) for d in [30, -30]]
    if Arm.num_actions == 8:
        dt0 = [round(math.radians(d), 4) for d in [-30, 10, 0, 80, -30, 10, 0, 80]]
        dt1 = [round(math.radians(d), 4) for d in [-30, 10, 0, 100, 30, 10, 0, 100]]
        dt2 = [round(math.radians(d), 4) for d in [30, 10, 0, 100, -30, 10, 0, 100]]
    if Arm.num_actions == 10:
        dt0 = [round(math.radians(d), 4) for d in [-30, 10, 0, 80, -100, -30, 10, 0, 80, -100]]
        dt1 = [round(math.radians(d), 4) for d in [-30, 10, 0, 100, -100, 30, 10, 0, 100, -100]]
        dt2 = [round(math.radians(d), 4) for d in [30, 10, 0, 100, -100, -30, 10, 0, 100, -100]]
    if Arm.num_actions == 14:
        dt0 = [round(math.radians(d), 4) for d in [-30, 10, 0, 80, -100, 0, 0, -30, 10, 0, 80, -100, 0, 0]]
        dt1 = [round(math.radians(d), 4) for d in [-30, 10, 0, 100, -100, 30, 10, 30, 10, 0, 100, -100, 30, 10]]
        dt2 = [round(math.radians(d), 4) for d in [30, 10, 0, 100, -100, -30, -10, -30, 10, 0, 100, -100, -30, -10]]
    # 执行关节规划
    for i in range(2):
        print("wave round %d" % (i * 2 + 1))
        if Arm.hands_enable:
            Arm.set_gripper([50] * Arm.num_actions)
        Arm.set_joint_path(period, dt1)
        print("wave round %d" % (i * 2 + 2))
        if Arm.hands_enable:
            Arm.set_gripper([5] * Arm.num_actions)
        Arm.set_joint_path(period, dt2)
    print("return to zero")
    Arm.set_joint_path(period, dt0)
    print("Test complete ...")
    Arm.disconnect()


if __name__ == "__main__":
    main()