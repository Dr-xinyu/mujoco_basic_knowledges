import time

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink
from scipy.spatial.transform import Rotation as R, Slerp

scene_path = "assets/kuka_iiwa_14/scene.xml"


left_target_position = np.array([0.4,0.3,0.4])
left_final_rotation = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
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
        "joint1": np.pi,
        "joint2": np.pi,
        "joint3": np.pi,
        "joint4": np.pi,
        "joint5": np.pi,
        "joint6": np.pi,
        "joint7": np.pi
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
    # wrist_3_geoms = mink.get_body_geom_ids(model, model.body("wrist_3_link").id)
    # collision_pairs = [
    #     (wrist_3_geoms, ["floor"]),
    # ]

    limits = [
        mink.ConfigurationLimit(model=configuration.model),
    ]

    velocity_limit = mink.VelocityLimit(model, max_velocities)
    limits.append(velocity_limit)
    ## =================== ##

    return tasks, limits, configuration


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    # keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    tasks, limits, configuration = setup_mink(model=model)
    initial_q = np.array([-0.186, 0.817, -0.1, -0.942, -1.1, -0.984, -0.428])

    linear_speed = 0.5

    data.qpos[:7] = initial_q

    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    site_pos = data.site_xpos[site_id].copy()
    site_r = data.site_xmat[site_id].copy().reshape(3, 3)

    se3_trajectories, actual_dt = generate_cartesian_trajectory(start_pos=site_pos, end_pos=left_target_position,
                                                                start_rotation_matrix=site_r, end_rotation_matrix=left_final_rotation,
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
                # rate.sleep()

        flag1 = 5
        while 1:
            if flag1 > 1:
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
                site_pos = data.site_xpos[site_id].copy()
                print('the site pos is ' + str(site_pos))
                flag1 -= 1
                time.sleep(0.002)
            viewer.sync()

