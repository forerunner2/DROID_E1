import logging
import mujoco
import mujoco.viewer
import numpy as np
from tqdm import tqdm
from Config import Config

_LOGGER = logging.getLogger(__name__)


class MujoBase:
    def __init__(self, _modeL_path, *, headless: bool = False):
        self.cfg = Config
        self.model = mujoco.MjModel.from_xml_path(filename=_modeL_path)
        self.num_joint = self.model.nu
        self.model.opt.timestep = self.cfg.dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)

        self._headless = headless
        self.viewer = None

        self.cfg.default_joints = np.zeros(self.num_joint, dtype=np.float32)
        self.cfg.dof_stiffness = np.zeros(self.num_joint, dtype=np.float32)
        self.cfg.dof_damping = np.zeros(self.num_joint, dtype=np.float32)
        self.cfg.effort_limit = np.zeros(self.num_joint, dtype=np.float32)
        self.cnt_pd_loop = 0

    def _maybe_create_viewer(self):
        if self._headless or self.viewer is not None:
            return
        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        except Exception as exc:  # pylint: disable=broad-except
            _LOGGER.warning("创建MuJoCo Viewer失败，将继续以headless模式运行：%s", exc)
            self.viewer = None
            self._headless = True

    def get_joint_names(self):
        actuators = []
        for i in range(0, self.num_joint):
            joint_id = self.model.actuator_trnid[i]  # 获取关节 ID
            _name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id[0])  # 获取关节名称
            actuators.append(_name)
        print("\nMujoco actuators order:\n",actuators, "\n")
        return actuators

    def get_robot_state(self):
        q = self.data.qpos.astype(np.float32)[7:]
        dq = self.data.qvel.astype(np.float32)[6:]
        ang_vel = self.data.qvel[3:6].astype(np.float32)
        quat = self.data.qpos[3:7].astype(np.float32)
        quat[:] = quat[[1, 2, 3, 0]]
        return q, dq, quat, ang_vel

    def set_robot_state(self, target_q, q, dq):  # mujoco关节顺序输入输出
        self.data.ctrl = np.clip(self.cfg.dof_stiffness * (target_q - q) - self.cfg.dof_damping * dq,
                                 -self.cfg.effort_limit, self.cfg.effort_limit)  # Clamp torques
        mujoco.mj_step(self.model, self.data)
        self._maybe_create_viewer()
        if self.viewer is not None:
            self.viewer.cam.lookat[:] = self.data.qpos.astype(np.float32)[0:3]
            self.viewer.sync()

    def run(self):
        self.cnt_pd_loop = 0
        duration_second = 0.01  # 单位:s
        for _ in tqdm(range(int(self.cfg.run_duration / duration_second)), desc="Simulating..."):
            q, dq, quat, ang_vel = self.get_robot_state()
            # if self.cnt_pd_loop % 10 == 0:
            # Generate PD control
            self.set_robot_state(0, q, dq)
            self.cnt_pd_loop += 1
        if self.viewer is not None:
            self.viewer.close()


if __name__ == '__main__':
    mybot = MujoBase()

    print("start main run")
    try:
        while True:
            mybot.run()
    except KeyboardInterrupt:
        print("\n用户中断，停止程序")
    finally:
        print("\n停止程序")
