import time

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

xml_path = "assets/universal_robots_ur5e/scene.xml"


def get_target_pose():
    target_pos = np.array([0.35, 0, 0.2])
    R_target = np.array([
        [1, 0, 0],  # X 轴：世界 X
        [0, -1, 0],  # Y 轴：世界 -Y（保持右手系）
        [0, 0, -1]  # Z 轴：世界 -Z（朝下）
    ])
    target_pose = mink.SE3.from_rotation_and_translation(
        rotation=mink.SO3.from_matrix(R_target),
        translation=target_pos

    )
    return target_pose


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
    target_pose = get_target_pose()
    end_effector_task = tasks[0]

    # IK settings.
    solver = "daqp"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 200

    rate = RateLimiter(frequency=100.0, warn=False)
    end_effector_task.set_target(target_pose)
    for _ in range(max_iters):
        vel = mink.solve_ik(configuration, tasks, rate.dt, solver=solver, limits=limits)
        configuration.integrate_inplace(vel, rate.dt)
        err = end_effector_task.compute_error(configuration)
        if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-3:
            print('the configuration q is '+ str(configuration.q))
            pose = configuration.get_transform_frame_to_world("attachment_site", "site")
            print("EE XYZ (mink):", pose.translation)
            break
