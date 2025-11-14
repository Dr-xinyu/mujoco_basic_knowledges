"""
=========================================
mink + MuJoCo：仅驱动机械臂（position）
=========================================
pip install mujoco mink numpy
"""
import mujoco
import mink
import numpy as np

# ---------- 1. 加载模型 ---------- #
xml_path = "your_scene.xml"          # 你的完整 MJCF（含桌子/物体）
model = mujoco.MjModel.from_xml_path(xml_path)
data  = mujoco.MjData(model)

# ---------- 2. 拿到“机械臂”actuator 索引 ---------- #
arm_act_names = [f"joint{i}_motor_ur5right" for i in range(6)] + \
                [f"joint{i}_motor_ur5left"  for i in range(6)]
arm_act_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                        for n in arm_act_names], dtype=np.int32)

# ---------- 3. 拿到机械臂关节在 qpos 里的地址 ---------- #
arm_jnt_names = [f"joint{i}_ur5right" for i in range(6)] + \
                [f"joint{i}_ur5left"  for i in range(6)]
arm_jnt_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                        for n in arm_jnt_names])
arm_qpos_adr = model.jnt_qposadr[arm_jnt_ids]  # 12 个地址

# ---------- 4. 初始目标 = 当前角度 ---------- #
q_target_arm = data.qpos[arm_qpos_adr].copy()

# ---------- 5. 配置 IK ---------- #
site_name = "ur_EE_ur5right"        # 先只跟踪右臂末端，左臂同理
task = mink.FrameTask(
    frame_name=site_name,
    frame_type="site",
    position_cost=1.0,
    orientation_cost=1.0,
)

limits = [
    mink.ConfigurationLimit(model=model),
    mink.VelocityLimit(np.full(model.nv, 2.0)),  # rad/s
]

dt = model.opt.timestep

# ---------- 6. 主循环 ---------- #
with mujoco.viewer.launch_passive(model, data) as viewer:
    target = np.array([0.5, 0.0, 0.3])  # 目标位置
    while viewer.is_running():
        # 6.1 更新任务
        task.set_target(mink.SE3.from_pos_quat(target, [1, 0, 0, 0]))

        # 6.2 速度 IK（只返回 12 个自由度）
        dq = mink.solve_ik(
            configuration=mink.Configuration(model, data),
            tasks=[task],
            constraints=limits,
            dt=dt,
            solver="quadprog",
        )[arm_qpos_adr]          # 只取机械臂部分

        # 6.3 积分 → 目标角度
        q_target_arm += dq * dt

        # 6.4 ****** 只写机械臂 actuator，其余通道不动 ******
        data.ctrl[arm_act_ids] = q_target_arm

        # 6.5 仿真步进
        mujoco.mj_step(model, data)
        viewer.sync()