import torch
import numpy as np

from extract.rl.agents.ac_agent import SACAgent, LangConditionedSACAgent
from extract.utils.general_utils import ParamDict, ConstantSchedule, AttrDict
from extract.utils.pytorch_utils import check_shape, map2torch


class ActionPriorSACAgent(SACAgent):
    """Implements SAC with non-uniform, learned action / skill prior."""

    def __init__(self, config):
        SACAgent.__init__(self, config)
        self._target_divergence = self._hp.td_schedule(self._hp.td_schedule_params)

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "alpha_min": 0,  # minimum value alpha is clipped to, no clipping if None
                "td_schedule": ConstantSchedule,  # schedule used for target divergence param
                "td_schedule_params": AttrDict(  # parameters for target divergence schedule
                    p=0.5,
                ),
            }
        )
        return super()._default_hparams().overwrite(default_dict)

    def sample_old_data(self):
        old_data_batch = next(self.old_data_buffer)
        # this will be a batch_size x dim tensor
        list_of_relevant_keys = [
            "states",
            "next_states",
            "actions",
            "reward",
            "done",
        ]
        # Make old_data_batch to attribute dict
        old_data_batch = AttrDict(old_data_batch)
        assert all([key in old_data_batch for key in list_of_relevant_keys])
        for key in list_of_relevant_keys:
            assert key in old_data_batch, f"Key {key} not in old data batch"
        # relabel the action with the encoder output, hacky only works for SPiRL and EXTRACT
        with torch.no_grad():
            # create the input dict with the observation, the low level actions,
            # TO device.
            for key, value in old_data_batch.items():
                old_data_batch[key] = value.to(self._hp.device)
            skills = self.policy.prior_net._run_inference(old_data_batch)
            if self._hp.rlpd_sample_old_encoded_actions:
                skills = skills.sample()
            else:
                skills = skills.mu
        # To CPU again.
        # for key, value in old_data_batch.items():
        #     old_data_batch[key] = value.detach().cpu().numpy()
        # skills = skills.detach().cpu().numpy()
        rl_old_data_batch = AttrDict(
            observation=old_data_batch["states"][:, 0],
            observation_next=old_data_batch["next_states"][:, 0],
            action=skills,
            reward=old_data_batch["reward"].squeeze(),
            done=old_data_batch["done"].squeeze(),
        )
        return rl_old_data_batch

    def update(self, experience_batch):
        info = super().update(experience_batch)
        info.target_divergence = self._target_divergence(self.schedule_steps)
        return info

    def _compute_alpha_loss(self, policy_output):
        """Computes loss for alpha update based on target divergence."""
        return (
            self.alpha
            * (
                self._target_divergence(self.schedule_steps)
                - policy_output.prior_divergence
            )
            .detach()
            .mean()
        )

    def _compute_policy_loss(self, experience_batch, policy_output):
        """Computes loss for policy update."""
        q_est = self._compute_critic_values(experience_batch, policy_output)
        policy_loss = -1 * q_est + self.alpha * policy_output.prior_divergence[:, None]
        check_shape(policy_loss, [self._hp.batch_size, 1])
        return policy_loss.mean()

    def _compute_next_value(self, experience_batch, policy_output):
        """Computes value of next state for target value computation."""
        q_next = self._compute_next_critic_values(experience_batch, policy_output)
        next_val = q_next - self.alpha * policy_output.prior_divergence[:, None]
        check_shape(next_val, [self._hp.batch_size, 1])
        return next_val.squeeze(-1)

    def _aux_info(self, experience_batch, policy_output):
        """Stores any additional values that should get logged to WandB."""
        aux_info = super()._aux_info(experience_batch, policy_output)
        aux_info.prior_divergence = policy_output.prior_divergence.mean()
        if (
            "ensemble_divergence" in policy_output
        ):  # when using ensemble thresholded prior divergence
            aux_info.ensemble_divergence = policy_output.ensemble_divergence.mean()
            aux_info.learned_prior_divergence = (
                policy_output.learned_prior_divergence.mean()
            )
            aux_info.below_ensemble_div_thresh = (
                policy_output.below_ensemble_div_thresh.mean()
            )
        return aux_info

    def state_dict(self, *args, **kwargs):
        d = super().state_dict(*args, **kwargs)
        d["update_steps"] = self._update_steps
        return d

    def load_state_dict(self, state_dict, *args, **kwargs):
        self._update_steps = state_dict.pop("update_steps")
        super().load_state_dict(state_dict, *args, **kwargs)

    @property
    def alpha(self):
        if self._hp.alpha_min is not None:
            return torch.clamp(super().alpha, min=self._hp.alpha_min)
        return super().alpha


class RandActScheduledActionPriorSACAgent(ActionPriorSACAgent):
    """Adds scheduled call to random action (aka prior execution) -> used if downstream policy trained from scratch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._omega = self._hp.omega_schedule(self._hp.omega_schedule_params)

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "omega_schedule": ConstantSchedule,  # schedule used for omega param
                "omega_schedule_params": AttrDict(  # parameters for omega schedule
                    p=0.1,
                ),
            }
        )
        return super()._default_hparams().overwrite(default_dict)

    def _act(self, obs, extra_info=None):
        """Call random action (aka prior policy) omega percent of times."""
        if np.random.rand() <= self._omega(self._update_steps):
            return super()._act_rand(obs, extra_info=extra_info)
        else:
            return super()._act(obs, extra_info=extra_info)

    def update(self, experience_batch):
        if (
            "delay" in self._hp.omega_schedule_params
            and self._update_steps < self._hp.omega_schedule_params.delay
        ):
            # if schedule has warmup phase in which *only* prior is sampled, train policy to minimize divergence
            self.replay_buffer.append(experience_batch)
            experience_batch = self.replay_buffer.sample(n_samples=self._hp.batch_size)
            experience_batch = map2torch(experience_batch, self._hp.device)
            policy_output = self._run_policy(experience_batch.observation)
            policy_loss = policy_output.prior_divergence.mean()
            self._perform_update(policy_loss, self.policy_opt, self.policy)
            self._update_steps += 1
            info = AttrDict(prior_divergence=policy_output.prior_divergence.mean())
        else:
            info = super().update(experience_batch)
        info.omega = self._omega(self._update_steps)
        return info


class LanguageConditionedActionPriorSACAgent(
    LangConditionedSACAgent, ActionPriorSACAgent
):
    def _compute_q_estimates(self, experience_batch):
        return LangConditionedSACAgent._compute_q_estimates(self, experience_batch)

    def _compute_critic_values(self, experience_batch, policy_output):
        return LangConditionedSACAgent._compute_critic_values(
            self, experience_batch, policy_output
        )

    def _compute_next_critic_values(self, experience_batch, policy_output):
        return LangConditionedSACAgent._compute_next_critic_values(
            self, experience_batch, policy_output
        )
