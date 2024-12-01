import numpy as np
from gym import spaces

import torch

# from furniture_bench.envs.furniture_bench_env import FurnitureBenchEnv
from extract.rl.envs.furniture_bench import FurnitureBenchEnv
from furniture_bench.robot.robot_state import filter_and_concat_robot_state
from furniture_bench.robot.panda import PandaError


class FurnitureBenchImageFeatureFlat(FurnitureBenchEnv):
    """
    Environment getting flatten states as observations.
    The shape of the observation: 2048 + 2048 + 3 + 4 + 1 = 4104, # The R3M features from wrist and front camera, ee-pos, ee-rot, gripper width.
    """

    def __init__(self, config):
        super().__init__(config)

        self.robot_state_dim = 8

        encoder_type = config["encoder_type"]

        if encoder_type == "r3m":
            from r3m import load_r3m

            self.layer = load_r3m("resnet50")
            self.embedding_dim = 2048
        elif encoder_type == "vip":
            from vip import load_vip

            self.layer = load_vip()
            self.embedding_dim = 1024
        self.layer.requires_grad_(False)
        self.layer.eval()

    @property
    def observation_space(self):
        return spaces.Box(
            -np.inf, np.inf, (self.embedding_dim * 2 + self.robot_state_dim,)
        )

    def _get_observation(self):
        """If successful, returns (obs, True); otherwise, returns (None, False)."""
        obs, panda_error = super()._get_observation()
        robot_state = obs["robot_state"]
        image1 = np.moveaxis(obs["color_image1"], -1, 0).astype(np.float32)
        image2 = np.moveaxis(obs["color_image2"], -1, 0).astype(np.float32)

        with torch.no_grad():
            image1 = self.layer(torch.tensor(image1, device="cuda").unsqueeze(0))
            image1 = image1.squeeze().detach().cpu().numpy()
            image2 = self.layer(torch.tensor(image2, device="cuda").unsqueeze(0))
            image2 = image2.squeeze().detach().cpu().numpy()

        return np.concatenate(
            [
                image1,
                image2,
                robot_state["ee_pos"],
                robot_state["ee_quat"],
                robot_state["gripper_width"],
            ],
            axis=0,
        ), panda_error

        # For reference: data conversion:
        #     observations = np.concatenate(
        #     [
        #         robot0_eye_in_hand_image_r3m_feature,
        #         agentview_image_r3m_feature,
        #         robot0_eef_pos,
        #         robot0_eef_quat,
        #         robot0_gripper_qpos,
        #     ],
        #     axis=1,
        # )
