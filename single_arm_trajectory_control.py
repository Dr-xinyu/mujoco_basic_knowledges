import time

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink
from scipy.spatial.transform import Rotation as R, Slerp

xml_path = "assets/universal_robots_ur5e/scene.xml"

end_pose = np.array([0.35, 0, 0.6])
r_end = np.array([
    [1, 0, 0],  # X 轴：世界 X
    [0, -1, 0],  # Y 轴：世界 -Y（保持右手系）
    [0, 0, -1]  # Z 轴：世界 -Z（朝下）
])


def generate_cartesian_trajectory(start_pos, end_pos,
                                  start_rotation_matrix,
                                  end_rotation_matrix,
                                  linear_speed,
                                  control_dt):
    displacement = end_pos - start_pos
    distance = np.linalg.norm(displacement)
    if distance < 1e-6:
        return [(start_pos.copy(), start_rotation_matrix.copy())], control_dt

    total_time = distance / linear_speed
    print('the total time is ' + str(total_time))
    num_points = max(2, int(np.ceil(total_time / control_dt)))
    actual_dt = total_time / (num_points - 1)

    # 线性插值
    positions = [start_pos + t * displacement for t in np.linspace(0, 1, num_points)]

    # 旋转的slerp 插值
    start_rot = R.from_matrix(start_rotation_matrix)
    end_rot = R.from_matrix(end_rotation_matrix)

    # start_q_xyzw = start_rot.as_quat()
    # end_q_xyzw = end_rot.as_quat()

    key_times = [0.0, 1.0]
    key_rots = R.concatenate([start_rot, end_rot])  # 或 from_quat([...])

    slerp = Slerp(key_times, key_rots)
    interp_times = np.linspace(0, 1, num_points)
    interp_rots = slerp(interp_times)

    quaternions = []
    for quat_xyzw in interp_rots.as_quat():  # 每个是 [x, y, z, w]
        x, y, z, w = quat_xyzw
        quat_wxyz = np.array([w, x, y, z])
        quaternions.append(quat_wxyz)

    se3_trajectory = []
    for pos, quaternion in zip(positions, quaternions):
        se3 = mink.SE3.from_rotation_and_translation(rotation=mink.SO3(wxyz=quaternion), translation=pos)
        se3_trajectory.append(se3)
    return se3_trajectory, actual_dt


def setup_mink(model):
    configuration = mink.Configuration(model)
    max_velocities = {
        "shoulder_pan": np.pi,
        "shoulder_lift": np.pi,
        "elbow": np.pi,
        "wrist_1": np.pi,
        "wrist_2": np.pi,
        "wrist_3": np.pi,
    }

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

    return tasks, limits, configuration


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    tasks, limits, configuration = setup_mink(model=model)
    initial_q = np.array([-2.45, -0.817, 1.32, -0.628, 0, -0.754])

    linear_speed = 0.5

    data.qpos[:6] = initial_q
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    site_pos = data.site_xpos[site_id].copy()
    site_r = data.site_xmat[site_id].copy().reshape(3, 3)

    se3_trajectories, actual_dt = generate_cartesian_trajectory(start_pos=site_pos, end_pos=end_pose,
                                                                start_rotation_matrix=site_r, end_rotation_matrix=r_end,
                                                                linear_speed=linear_speed, control_dt=1 / 60)

    end_effector_task = tasks[0]

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
        # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD

        # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY

        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        rate = RateLimiter(frequency=60.0, warn=False)

        for se3_trjectory in se3_trajectories:

            end_effector_task.set_target(se3_trjectory)

            reached = False
            while not reached:
                configuration.update(data.qpos)
                vel = mink.solve_ik(configuration, tasks, rate.dt, solver=solver, limits=limits)
                configuration.integrate_inplace(vel, rate.dt)
                err = end_effector_task.compute_error(configuration)
                if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-3:
                    # print(configuration.q)
                    reached = True

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
                rate.sleep()

        # flag = 5
        # while 1:
        #     if flag > 1:
        #         site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        #         site_pos = data.site_xpos[site_id].copy()
        #         print('the site pos is ' + str(site_pos))
        #         flag -= 1
        #         time.sleep(0.002)
        #     viewer.sync()
