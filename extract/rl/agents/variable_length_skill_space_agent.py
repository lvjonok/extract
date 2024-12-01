import contextlib

import numpy as np
from collections import defaultdict

from extract.rl.agents.skill_space_agent import SkillSpaceAgent
from matplotlib.backends.backend_agg import FigureCanvasAgg
from extract.utils.general_utils import (
    AttrDict,
    MaxSizeQueue,
)
from extract.utils.pytorch_utils import map2torch, map2np, no_batchnorm_update
from extract.utils.vis_utils import get_plot_object  # , get_double_figure
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class VariableLengthSkillSpaceAgent(SkillSpaceAgent):
    """Agent that acts based on pre-trained VAE skill decoder, and also saves skill lengths."""

    def __init__(self, config):
        super().__init__(config)
        # self.skill_lengths = MaxSizeQueue(10000)
        self.skill_cluster_assignments = defaultdict(list)
        self.extra_info = AttrDict()

    def _act(self, obs, extra_info):
        assert (
            len(obs.shape) == 2 and obs.shape[0] == 1
        )  # assume single-observation batches with leading 1-dim
        if not self.action_plan:
            # generate action plan if the current one is empty
            inputs = self._split_obs(obs)
            with (
                no_batchnorm_update(self._policy)
                if obs.shape[0] == 1
                else contextlib.suppress()
            ):
                if extra_info is not None and "lang" in extra_info:
                    inputs.lang = extra_info.lang
                self.action_plan = self._policy.run(inputs)
            discrete_skill = inputs.skills.squeeze(0).item()
            self.skill_cluster_assignments[discrete_skill].append(len(self.action_plan))
            self.extra_info.discrete_skill_choice = discrete_skill

        return AttrDict(action=np.expand_dims(self.action_plan.popleft(), 0))

    def add_extra_env_info(self):
        extra_env_info = super().add_extra_env_info()
        extra_env_info.update(self.extra_info)
        return extra_env_info

    def visualize(self, logger, rollout_storage, step):
        # log skill lengths to replay buffer to visualize
        super().visualize(logger, rollout_storage, step)
        sns.set()
        fig, ax = plt.subplots(3, 1, figsize=(12, 14))
        all_lengths = np.array(
            [
                length
                for lengths in self.skill_cluster_assignments.values()
                for length in lengths
            ]
        )
        max_length = np.max(all_lengths)
        canvas = FigureCanvasAgg(fig)
        ax[0].hist(
            all_lengths,
            bins=np.arange(1, max_length + 1),
            density=True,
        )
        ax[0].set_title(f"Histogram of Skill Lengths")
        ordered_labels = list(range(self._hp.model_params.n_skills))
        ordered_values = [self.skill_cluster_assignments[key] for key in ordered_labels]
        ax[1].boxplot(
            ordered_values, labels=ordered_labels, showmeans=True, meanline=True
        )
        ax[1].set_title(
            f"Avg Skill Lengths: {np.mean([length for skill in self.skill_cluster_assignments for length in self.skill_cluster_assignments[skill]]):0.1f}"
        )

        # first make a dataframe
        skill_lengths_df = pd.DataFrame()
        for skill, lengths in self.skill_cluster_assignments.items():
            skill_lengths_df = pd.concat(
                (
                    skill_lengths_df,
                    pd.DataFrame(
                        {
                            "skill": [skill for _ in range(len(lengths))],
                            "length": lengths,
                        }
                    ),
                ),
                ignore_index=True,
            )
        # now plot a simple histogram over which skills were called how many times, ensure that the number of bins are equal to the number of skills so spacing isn't messed up
        sns.histplot(
            data=skill_lengths_df,
            x="skill",
            hue="skill",
            element="step",
            ax=ax[2],
            bins=self._hp.model_params.n_skills,
        )
        ax[2].set_title("Skill Counts")
        logger.log_plot(fig, "skill_lengths", step)
        logger.log_scalar_dict(
            {
                "avg_skill_length_values": np.mean(all_lengths),
                "skill_length_histogram": all_lengths,
            },
            step=step,
        )
        self.skill_cluster_assignments = defaultdict(list)

        return get_plot_object(fig, ax, canvas)

    def _split_obs(self, obs):
        assert obs.shape[1] == self._policy.state_dim + self._policy.latent_dim
        out = AttrDict(
            states=obs[:, : -self._policy.latent_dim],  # condition decoding on state
            z=obs[:, -self._policy.latent_dim + 1 :],
            skills=obs[:, -self._policy.latent_dim : -self._policy.latent_dim + 1],
        )
        return out


class ClosedLoopVariableLengthSkillSpaceAgent(VariableLengthSkillSpaceAgent):
    def __init__(self, config):
        super().__init__(config)
        self.reset()

    def reset(self):
        super().reset()
        self._last_skill = None
        self._last_z = None
        self._was_last_action = True
        self._time_since_last_action = 0

    @property
    def _need_new_hl_action(self):
        return (
            self._was_last_action
            or self._time_since_last_action >= self._policy._hp.max_rollout_steps
        )

    def _act(self, obs, extra_info):
        assert (
            len(obs.shape) == 2 and obs.shape[0] == 1
        )  # assume single-observation batches with leading 1-dim
        inputs = self._split_obs(obs)
        if self._need_new_hl_action:
            # new skill
            self._last_skill = inputs.skills
            self._last_z = inputs.z
            discrete_skill = inputs.skills.squeeze(0).item()
            self.extra_info.discrete_skill_choice = discrete_skill
            self.skill_cluster_assignments[discrete_skill].append(
                self._time_since_last_action
            )
            self._time_since_last_action = 0
        with (
            no_batchnorm_update(self._policy)
            if obs.shape[0] == 1
            else contextlib.suppress()
        ):
            inputs.z = self._last_z
            inputs.skills = self._last_skill
            self.action_plan, self._was_last_action = self._policy.run(inputs)
        action = self.action_plan[0]
        self._time_since_last_action += 1
        if self._need_new_hl_action:
            self.action_plan.popleft()  # this empties the action plan so the higher level policy can take an action
        return AttrDict(action=np.expand_dims(action, 0))


class VariableLengthACSkillSpaceAgent(VariableLengthSkillSpaceAgent):
    """Unflattens prior input part of observation."""

    def __init__(self, config):
        super().__init__(config)

    def _split_obs(self, obs):
        unflattened_obs = map2np(
            self._policy.unflatten_obs(
                map2torch(obs[:, : -self._policy.latent_dim], device=self.device)
            )
        )
        return AttrDict(
            images=unflattened_obs,
            z=obs[:, -self._policy.latent_dim + 1 :],
            skills=obs[:, -self._policy.latent_dim : -self._policy.latent_dim + 1],
        )


class ClosedLoopVariableLengthACSKillSpaceAgent(
    ClosedLoopVariableLengthSkillSpaceAgent
):
    def _split_obs(self, obs):
        unflattened_obs = self._policy.unflatten_obs(
            map2torch(obs[:, : -self._policy.latent_dim], device=self.device)
        )
        return AttrDict(
            states=self._policy.enc_obs(unflattened_obs.prior_obs),
            z=obs[:, -self._policy.latent_dim + 1 :],
            skills=obs[:, -self._policy.latent_dim : -self._policy.latent_dim + 1],
        )
