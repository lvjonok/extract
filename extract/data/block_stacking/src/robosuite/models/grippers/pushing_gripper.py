"""
A version of TwoFingerGripper but always closed.
"""
import numpy as np
from extract.data.block_stacking.src.robosuite.models.grippers.two_finger_gripper import (
    TwoFingerGripper,
)


class PushingGripper(TwoFingerGripper):
    """
    Same as TwoFingerGripper, but always closed
    """

    def format_action(self, action):
        return np.array([1, -1])

    @property
    def dof(self):
        return 1
