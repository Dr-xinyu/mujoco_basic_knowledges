import time

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink
from scipy.spatial.transform import Rotation as R, Slerp

xml_path = "assets/kuka_iiwa_14/iiwa14.xml"
scene_path = "assets/kuka_iiwa_14/scene.xml"


# def construct_dual_arm_model() -> mujoco.MjModel:
#     root = mujoco.MjSpec()
#     root.stat.meansize = 0.08
#     root.stat.extent = 1.0
#     root.stat.center = (0, 0, 0.5)
#     root.visual.global_.azimuth = -180
#     root.visual.global_.elevation = -20
#
#     root.worldbody.add_light(pos=(0, 0, 1.5), directional=True)
#
#     origin_site = root.worldbody.add_site(pos=[0, 0, 0], group=6)
#     scene_spec = mujoco.MjSpec.from_file(scene_path)
#     root.attach(scene_spec, site=origin_site)
#
#     left_site = root.worldbody.add_site(name="l_attachment_site", pos=[0, 0.5, 0], group=5)
#     right_site = root.worldbody.add_site(name="r_attachment_site", pos=[0, -0.5, 0], group=5)
#
#     left_iiwa = mujoco.MjSpec.from_file(xml_path)
#     left_iiwa.modelname = "l_iiwa"
#     left_iiwa.key("home").delete()
#     for i in range(len(left_iiwa.geoms)):
#         left_iiwa.geoms[i].name = f"geom_{i}"
#     root.attach(left_iiwa, site=left_site, prefix="l_iiwa/")
#
#     right_iiwa = mujoco.MjSpec.from_file(xml_path)
#     right_iiwa.modelname = "r_iiwa"
#     right_iiwa.key("home").delete()
#     for i in range(len(right_iiwa.geoms)):
#         right_iiwa.geoms[i].name = f"geom_{i}"
#     root.attach(right_iiwa, site=right_site, prefix="r_iiwa/")
#
#     return root.compile()


def mirror_iiwa_joints(q_left: np.ndarray) -> np.ndarray:
    """将左臂关节角映射为右臂镜像关节角（适用于 Y 轴对称安装的 iiwa）"""
    if q_left.shape[-1] != 7:
        raise ValueError("Expected 7-DOF joint angles.")
    q_right = q_left.copy()
    # 偶数索引关节取负：0,2,4,6（绕 Z 轴的关节）
    q_right[..., [0, 2, 4, 6]] *= -1
    return q_right


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


if __name__ == "__main__":
    model = construct_dual_arm_model()

    configuration = mink.Configuration(model)

    site_point_names = ["l_iiwa/attachment_site", "r_iiwa/attachment_site"]
    left_init_q = np.array([-0.186, 0.817, -0.1, -0.942, -1.1, -0.984, -0.428])
    right_init_q = mirror_iiwa_joints(left_init_q)

    right_target_position = np.array([0.5, -0.7, 0.4])
    left_target_position = np.array([0.5, 0.7, 0.4])

    target_positions = [left_target_position, right_target_position]

    right_final_rotation = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])
    left_final_rotation = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])

    target_rotations = [left_final_rotation, right_final_rotation]
    # model = mujoco.MjModel.from_xml_path(xml_path)

    data = mujoco.MjData(model)

    # print(data.qpos)
    data.qpos[:7] = right_init_q
    data.qpos[7:14] = left_init_q
    mujoco.mj_forward(model, data)

    # mujoco.viewer.launch(model, data)

    start_positions = []
    start_rotations = []
    for site_point_name in site_point_names:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_point_name)
        site_pos = data.site_xpos[site_id].copy()
        site_r = data.site_xmat[site_id].copy().reshape(3, 3)
        start_positions.append(site_pos)
        start_rotations.append(site_r)

    linear_speed = 0.3

    se3_all_arms_trajectories = []
    for i in range(len(site_point_names)):
        start_position = start_positions[i]
        start_rotation = start_rotations[i]
        end_position = target_positions[i]
        end_rotation = target_rotations[i]
        print("the start position is" + str(start_position))
        se3_trajectories, actual_dt = generate_cartesian_trajectory(start_pos=start_position, end_pos=end_position,
                                                                    start_rotation_matrix=start_rotation,
                                                                    end_rotation_matrix=start_rotation,
                                                                    linear_speed=linear_speed, control_dt=1 / 60)
        se3_all_arms_trajectories.append(se3_trajectories)

    tasks = [
        left_ee_task := mink.FrameTask(
            frame_name="l_iiwa/attachment_site",
            frame_type="site",
            position_cost=2.0,
            orientation_cost=1.0,
        ),
        right_ee_task := mink.FrameTask(
            frame_name="r_iiwa/attachment_site",
            frame_type="site",
            position_cost=2.0,
            orientation_cost=1.0,
        ),
        posture_task := mink.PostureTask(model, cost=1e-3),
    ]
    posture_task.set_target(configuration.q)

    max_velocities = {
        "l_iiwa/joint1": np.pi,
        "l_iiwa/joint2": np.pi,
        "l_iiwa/joint3": np.pi,
        "l_iiwa/joint4": np.pi,
        "l_iiwa/joint5": np.pi,
        "l_iiwa/joint6": np.pi,
        "l_iiwa/joint7": np.pi,
        "r_iiwa/joint1": np.pi,
        "r_iiwa/joint2": np.pi,
        "r_iiwa/joint3": np.pi,
        "r_iiwa/joint4": np.pi,
        "r_iiwa/joint5": np.pi,
        "r_iiwa/joint6": np.pi,
        "r_iiwa/joint7": np.pi,
    }

    limits = [
        mink.ConfigurationLimit(model=configuration.model),
    ]

    velocity_limit = mink.VelocityLimit(model, max_velocities)
    limits.append(velocity_limit)
    solver = "daqp"

    with mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
    ) as viewer:

        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        rate = RateLimiter(frequency=60.0, warn=False)
        viewer.sync()

        for se3_left_trajectory, se3_right_trajectory in zip(*se3_all_arms_trajectories):
            print()

        for se3_left_trajectory, se3_right_trajectory in zip(*se3_all_arms_trajectories):
            left_ee_task.set_target(se3_left_trajectory)
            right_ee_task.set_target(se3_right_trajectory)

            reached = 0
            while reached < 3:
                configuration.update(data.qpos)
                vel = mink.solve_ik(
                    configuration, tasks, rate.dt, solver, 1e-3, limits=limits
                )
                configuration.integrate_inplace(vel, rate.dt)

                err1 = left_ee_task.compute_error(configuration)
                err2 = right_ee_task.compute_error(configuration)
                # if np.linalg.norm(err1[:3]) < 1e-4 and np.linalg.norm(err1[3:]) < 1e-3 \
                #         and np.linalg.norm(err2[:3]) < 1e-4 and np.linalg.norm(err2[3:] < 1e-3):
                #     #     # print(configuration.q)
                #     reached = True

                data.ctrl = configuration.q
                mujoco.mj_step(model, data)
                viewer.sync()
                rate.sleep()
                reached += 1
