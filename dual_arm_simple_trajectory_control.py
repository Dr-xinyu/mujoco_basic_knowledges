import time

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink
from scipy.spatial.transform import Rotation as R, Slerp

xml_path = "assets/kuka_iiwa_14/iiwa14.xml"
scene_path = "assets/kuka_iiwa_14/scene.xml"


def construct_dual_arm_model() -> mujoco.MjModel:
    root = mujoco.MjSpec()
    root.stat.meansize = 0.08
    root.stat.extent = 1.0
    root.stat.center = (0, 0, 0.5)
    root.visual.global_.azimuth = -180
    root.visual.global_.elevation = -20

    root.worldbody.add_light(pos=(0, 0, 1.5), directional=True)

    origin_site = root.worldbody.add_site(pos=[0, 0, 0], group=6)
    scene_spec = mujoco.MjSpec.from_file(scene_path)
    root.attach(scene_spec, site = origin_site)

    left_site = root.worldbody.add_site(name="l_attachment_site", pos=[0, 0.5, 0], group=5)
    right_site = root.worldbody.add_site(name="r_attachment_site", pos=[0, -0.5, 0], group=5)

    left_iiwa = mujoco.MjSpec.from_file(xml_path)
    left_iiwa.modelname = "l_iiwa"
    left_iiwa.key("home").delete()
    for i in range(len(left_iiwa.geoms)):
        left_iiwa.geoms[i].name = f"geom_{i}"
    root.attach(left_iiwa, site=left_site, prefix="l_iiwa/")

    right_iiwa = mujoco.MjSpec.from_file(xml_path)
    right_iiwa.modelname = "r_iiwa"
    right_iiwa.key("home").delete()
    for i in range(len(right_iiwa.geoms)):
        right_iiwa.geoms[i].name = f"geom_{i}"
    root.attach(right_iiwa, site=right_site, prefix="r_iiwa/")



    return root.compile()


if __name__ == "__main__":
    # model = construct_dual_arm_model()
    left_init_q = np.array([0.086, 0.817, -0.1, -0.942, -1.1, -0.984, -0.428])
    right_init_q = np.array([-0.086, 0.817, -0.1, -0.942, -1.1, -0.984, -0.428])
    model = mujoco.MjModel.from_xml_path(xml_path)

    data = mujoco.MjData(model)

    # print(data.qpos)
    # data.qpos[:7] = left_init_q
    # data.qpos[7:14] = right_init_q
    # mujoco.mj_forward(model, data)

    # keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.viewer.launch(model, data)

    # with mujoco.viewer.launch_passive(
    #         model=model,
    #         data=data,
    #         show_left_ui=False,
    #         show_right_ui=False,
    # ) as viewer:
    #     mujoco.mjv_defaultFreeCamera(model, viewer.cam)
    #
    #     while viewer.is_running():
    #         viewer.sync()