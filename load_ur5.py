import time

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

    # Initialize key_callback function.
    mujoco.viewer.launch(model, data)
