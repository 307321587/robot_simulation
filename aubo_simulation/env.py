import time
import math
import random

import numpy as np
import pybullet as p
import pybullet_data

from .utilities import Camera
from collections import namedtuple
from attrdict import AttrDict
from tqdm import tqdm


class FailToReachTargetError(RuntimeError):
    pass


class AuboEnv:

    SIMULATION_STEP_DELAY = 1 / 240.

    def __init__(self, robot, camera=None, vis=False) -> None:
        self.robot = robot
        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)
        self.camera = camera
        
        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeID = p.loadURDF("plane.urdf")

        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        # custom sliders to tune parameters (name of the parameter,range,initial value)
        self.xin = p.addUserDebugParameter("x", -1, 1, 0.000508)
        self.yin = p.addUserDebugParameter("y", -1, 1, -0.2277451)
        self.zin = p.addUserDebugParameter("z", 0, 2, 1.130844)
        self.rollId = p.addUserDebugParameter("roll", -3.14, 3.14, np.pi/2)
        self.pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, 0)
        self.yawId = p.addUserDebugParameter("yaw", -np.pi/2, np.pi/2, 0)
        # For calculating the reward
        self.box_opened = False
        self.btn_pressed = False
        self.box_closed = False

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)

    def read_debug_parameter(self):
        # read the value of task parameter
        x = p.readUserDebugParameter(self.xin)
        y = p.readUserDebugParameter(self.yin)
        z = p.readUserDebugParameter(self.zin)
        roll = p.readUserDebugParameter(self.rollId)
        pitch = p.readUserDebugParameter(self.pitchId)
        yaw = p.readUserDebugParameter(self.yawId)

        gripper_opening_length = 0
        return x, y, z, roll, pitch, yaw,gripper_opening_length

    def step(self, action, control_method='joint'):
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """
        assert control_method in ('joint', 'end')
        self.robot.move_ee(action[:-1], control_method)
        self.robot.move_gripper(action[-1])
        for _ in range(120):  # Wait for a few steps
            self.step_simulation()

        return self.get_observation()

    def get_observation(self):
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs.update(dict(rgb=rgb, depth=depth, seg=seg))
        else:
            assert self.camera is None
        obs.update(self.robot.get_joint_obs())

        return obs

    def reset(self):
        self.robot.reset()
        end_pos, end_orn = self.robot.get_ee_pose()
        p.removeAllUserParameters(self.physicsClient)
        self.xin = p.addUserDebugParameter("x", -1, 1, end_pos[0])
        self.yin = p.addUserDebugParameter("y", -1, 1, end_pos[1])
        self.zin = p.addUserDebugParameter("z", 0, 2, end_pos[2])
        self.rollId = p.addUserDebugParameter("roll", -3.14, 3.14, end_orn[0])
        self.pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, end_orn[1])
        self.yawId = p.addUserDebugParameter("yaw", -np.pi/2, np.pi/2, end_orn[2])
        return np.array(end_pos), np.array(end_orn)

    def reset_camera(self,camera):
        self.camera=camera

    def close(self):
        p.disconnect(self.physicsClient)
