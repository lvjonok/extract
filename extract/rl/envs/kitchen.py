import numpy as np
from collections import defaultdict


from extract.utils.general_utils import AttrDict
from extract.utils.general_utils import ParamDict
from extract.rl.components.environment import GymEnv


class KitchenEnv(GymEnv):
    import d4rl

    """Tiny wrapper around GymEnv for Kitchen tasks."""

    SUBTASKS = [
        "microwave",
        "kettle",
        "slide cabinet",
        "hinge cabinet",
        "bottom burner",
        "light switch",
        "top burner",
    ]

    def reset_model_state(self, init_state):
        # TODO: figure this out
        self._env.robot_noise_ratio = 0
        # for generating video demonstrations
        reset_pos = init_state[:30].copy()
        reset_vel = init_state[30:-1].copy()
        self._env.robot.reset(self, reset_pos, reset_vel)
        self.goal = self._env._get_task_goal()
        return self._env._get_obs()

    def _default_hparams(self):
        return (
            super()
            ._default_hparams()
            .overwrite(
                ParamDict(
                    {
                        "name": "kitchen-mixed-v0",
                    }
                )
            )
        )

    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        return (
            obs,
            np.float64(rew),
            done,
            self._postprocess_info(info),
        )  # casting reward to float64 is important for getting shape later

    def reset(self):
        self.solved_subtasks = defaultdict(lambda: 0)
        return super().reset()

    def get_episode_info(self):
        info = super().get_episode_info()
        info.update(AttrDict(self.solved_subtasks))
        return info

    def _postprocess_info(self, info):
        """Sorts solved subtasks into separately logged elements."""
        completed_subtasks = info.pop("completed_tasks")
        for task in self.SUBTASKS:
            self.solved_subtasks[task] = (
                1 if task in completed_subtasks or self.solved_subtasks[task] else 0
            )
        if all(self.solved_subtasks.values()):
            info["success"] = True
        return info

    def render(self, mode=None):
        # TODO make env render in the correct size instead of downsizing after for performance
        if mode == "rgb_array":
            return (
                self._render_raw(
                    mode=mode,
                    width=self._hp.screen_width,
                    height=self._hp.screen_height,
                )
                / 255.0
            )  # render in the correct size for the agent
        img = self._render_raw(
            mode="rgb_array"
        )  # render in default size for video recording
        return img / 255.0

    def _render_raw(self, mode, width=None, height=None):
        """Returns rendering as uint8 in range [0...255]"""
        if width is not None and height is not None:
            return self._env.render(mode=mode, width=width, height=height)
        return self._env.render(mode=mode)

    @property
    def num_eval_tasks(self):
        """Returns number of evaluation tasks. Defaults to None if not set. Used for eval loop."""
        return 5


class NoGoalKitchenEnv(KitchenEnv):
    """Splits off goal from obs."""

    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        obs = obs[: int(obs.shape[0] / 2)]
        return obs, rew, done, info

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        return obs[: int(obs.shape[0] / 2)]
