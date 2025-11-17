import time

import imageio
import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

xml_path = "assets/universal_robots_ur5e/scene.xml"

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")



    ## =================== ##
    ## Setup IK.
    ## =================== ##

    configuration = mink.Configuration(model)
    max_velocities = {
        "shoulder_pan": np.pi,
        "shoulder_lift": np.pi,
        "elbow": np.pi,
        "wrist_1": np.pi,
        "wrist_2": np.pi,
        "wrist_3": np.pi,
    }

    print("Initial joint configuration (q0):", configuration.q)
    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1e-3,
        ),
        posture_task := mink.PostureTask(model, cost=1e-3),
    ]
    posture_task.set_target(configuration.q)

    # Enable collision avoidance between (wrist3, floor) and (wrist3, wall).
    wrist_3_geoms = mink.get_body_geom_ids(model, model.body("wrist_3_link").id)
    collision_pairs = [
        (wrist_3_geoms, ["floor"]),
    ]

    limits = [
        mink.ConfigurationLimit(model=configuration.model),
        mink.CollisionAvoidanceLimit(
            model=configuration.model,
            geom_pairs=collision_pairs,
        ),
    ]

    velocity_limit = mink.VelocityLimit(model, max_velocities)
    limits.append(velocity_limit)

    ## =================== ##

    mid = model.body("target").mocapid[0]
    target_pos = np.array([0.3, 0, 0.3])
    R_target = np.array([
        [1, 0, 0],  # X 轴：世界 X
        [0, -1, 0],  # Y 轴：世界 -Y（保持右手系）
        [0, 0, -1]  # Z 轴：世界 -Z（朝下）
    ])
    target_pose = mink.SE3.from_rotation_and_translation(
        rotation=mink.SO3.from_matrix(R_target),
        translation=target_pos

    )

    # IK settings.
    solver = "daqp"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    # Initialize key_callback function.
    with mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
    ) as viewer:
        # 打开世界坐标的x y z轴
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD

        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        rate = RateLimiter(frequency=100.0, warn=False)

        data.qpos[:] = model.key_qpos[keyframe_id]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

        # 先让机械臂到达目标
        reached = False
        images = []
        while viewer.is_running() and not reached:

            configuration.update(data.qpos)
            end_effector_task.set_target(target_pose)

            for _ in range(1):
                vel = mink.solve_ik(configuration, tasks, rate.dt, solver=solver, limits=limits)
                configuration.integrate_inplace(vel, rate.dt)
                err = end_effector_task.compute_error(configuration)
                if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-3:
                    reached = True
                    # print('finish')

            # 不使用驱动，只是看mink的ik是否准确
            # data.qpos = configuration.q
            # mujoco.mj_forward(model,data)

            # 使用驱动，还要看pd控制器是否准确
            data.ctrl = configuration.q
            mujoco.mj_step(model, data)

            # 打印mink求解的位置和当前的site实际的位置
            # pose = configuration.get_transform_frame_to_world("attachment_site", "site")
            # print("EE XYZ (mink):", pose.translation)
            # site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
            # site_pos = data.site_xpos[site_id].copy()
            # print('the site pos is '+ str(site_pos))
            viewer.sync()
            img = viewer.read_pixels()
            rate.sleep()

        imageio.mimsave("output.mp4", images, fps=60, codec="libx264")