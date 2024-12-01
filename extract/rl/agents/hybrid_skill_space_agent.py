from extract.rl.agents.prior_sac_agent import ActionPriorSACAgent
from extract.utils.general_utils import ParamDict, ConstantSchedule, AttrDict
from extract.utils.pytorch_utils import TensorModule, check_shape

import numpy as np
import torch


class HybridSkillSpaceActionPriorSACAgent(ActionPriorSACAgent):
    """Implements semantic imitation algorithm with SkiLD imitation on semantic latent and SPiRL on execution latent."""

    def __init__(self, *args, **kwargs):
        ActionPriorSACAgent.__init__(self, *args, **kwargs)
        self._discrete_target_divergence = self._hp.tdd_schedule(
            self._hp.tdd_schedule_params
        )

        # define domain divergence multiplier alpha_d
        if self._hp.fixed_alpha_d is not None:
            self._log_alpha_d = TensorModule(
                np.log(self._hp.fixed_alpha_d)
                * torch.ones(1, requires_grad=False, device=self._hp.device)
            )
        else:
            self._log_alpha_d = TensorModule(
                torch.zeros(1, requires_grad=True, device=self._hp.device)
            )
            self.alpha_d_opt = self._get_optimizer(
                self._hp.optimizer, self._log_alpha_d, self._hp.alpha_lr
            )

    def _default_hparams(self):
        return ActionPriorSACAgent._default_hparams(self).overwrite(
            ParamDict(
                {
                    "tdd_schedule": ConstantSchedule,  # schedule used for discrete target divergence param
                    "tdd_schedule_params": AttrDict(  # parameters for discrete target divergence schedule
                        p=0.5,
                    ),
                    "fixed_alpha_d": None,
                }
            )
        )

    def _update_continuous_alpha(self, experience_batch, policy_output):
        if self._hp.fixed_alpha is not None:
            return 0.0
        return (
            self.alpha
            * (
                self._target_divergence(self.schedule_steps)
                - policy_output.continuous_prior_divergence
            )
            .detach()
            .mean()
        )

    def _update_alpha(self, experience_batch, policy_output):
        if self._hp.fixed_alpha is None:
            alpha_loss = self._update_continuous_alpha(experience_batch, policy_output)
            self._perform_update(alpha_loss, self.alpha_opt, self._log_alpha)
        else:
            alpha_loss = 0.0

        # update alpha_d
        if self._hp.fixed_alpha_d is None:
            self.alpha_d_loss = (
                self.alpha_d
                * (
                    self._discrete_target_divergence(self.schedule_steps)
                    - policy_output.discrete_prior_divergence
                )
                .detach()
                .mean()
            )
            self._perform_update(self.alpha_d_loss, self.alpha_d_opt, self._log_alpha_d)
        else:
            self.alpha_d_loss = 0.0

        return alpha_loss

    def _aux_info(self, experience_batch, policy_output):
        # aux_info = ActionPriorSACAgent._aux_info(self, experience_batch, policy_output)
        aux_info = AttrDict()
        aux_info.update(
            AttrDict(
                discrete_prior_divergence=policy_output.discrete_prior_divergence.mean(),
                continuous_prior_divergence=policy_output.continuous_prior_divergence.mean(),
                alpha_d_loss=self.alpha_d_loss,
                alpha_d=self.alpha_d,
            )
        )
        return aux_info

    def state_dict(self, *args, **kwargs):
        d = ActionPriorSACAgent.state_dict(self)
        if hasattr(self, "alpha_d_opt"):
            d["alpha_d_opt"] = self.alpha_d_opt.state_dict()
        return d

    def load_state_dict(self, state_dict, *args, **kwargs):
        if "alpha_d_opt" in state_dict:
            self.alpha_d_opt.load_state_dict(state_dict.pop("alpha_d_opt"))
        ActionPriorSACAgent.load_state_dict(self, state_dict, *args, **kwargs)

    @property
    def alpha_d(self):
        if self._hp.alpha_min is not None:
            return torch.clamp(self._log_alpha_d().exp(), min=self._hp.alpha_min)
        return self._log_alpha_d().exp()

    def _prep_action(self, action):
        """Preprocessing of action to remove the discrete action from it before passing to the critic"""
        return action.float()[:, 1:]

    def _compute_policy_loss(self, experience_batch, policy_output):
        """Apply SAC+SPiRL loss to both discrete and continuous action policy."""
        q_est = self._compute_critic_values(experience_batch, policy_output)
        weighted_continuous_prior_divergence = (
            self.alpha * policy_output.continuous_prior_divergence[:, None]
        )
        discrete_divergence = (
            self.alpha_d * policy_output.discrete_prior_divergence[:, None]
        )

        policy_loss = (
            (-q_est + weighted_continuous_prior_divergence + discrete_divergence)
            * policy_output.prob_d
        ).sum(-1, keepdim=True)
        # + weighted_continuous_prior_divergence
        # + discrete_divergence

        check_shape(policy_loss, [self._hp.batch_size, 1])
        return policy_loss.mean()

    def _compute_next_value(self, experience_batch, policy_output):
        q_next = self._compute_next_critic_values(experience_batch, policy_output)
        weighted_continuous_prior_divergence = (
            self.alpha * policy_output.continuous_prior_divergence[:, None]
        )
        discrete_divergence = (
            self.alpha_d * policy_output.discrete_prior_divergence[:, None]
        )

        next_val = (
            policy_output.prob_d
            * (q_next - weighted_continuous_prior_divergence - discrete_divergence)
        ).sum(-1)
        check_shape(next_val, [self._hp.batch_size])
        return next_val

    def _compute_critic_loss(self, experience_batch, q_target, policy_output):
        qs = self._compute_q_estimates(experience_batch)
        if len(qs[0].shape) > 1: # multiple discrete skills
            qs = [q.gather(-1, policy_output.action_d.long()).squeeze(-1) for q in qs]
        check_shape(qs[0], [self._hp.batch_size])
        critic_losses = [0.5 * (q - q_target).pow(2).mean() for q in qs]
        return critic_losses, qs


class LangConditionedHybridSkillSpaceAgent(HybridSkillSpaceActionPriorSACAgent):
    def _compute_critic_values(self, experience_batch, policy_output):
        """Computes values for all critics."""
        operation = torch.mean if self._hp.critic_mean_for_policy_loss else torch.min
        q_est = operation(
            *[
                critic(
                    experience_batch.observation,
                    self._prep_action(policy_output.action),
                    experience_batch.lang,
                ).q
                for critic in self.critics
            ]
        )
        return q_est

    def _compute_next_critic_values(self, experience_batch, policy_output):
        """Computes values for all critics."""
        if self._num_critics == self._num_critic_subset:
            critic_targets = self.critic_targets
        else:
            critic_targets = self.critic_targets[
                np.random.choice(
                    self._num_critics, self._num_critic_subset, replace=False
                )
            ]
            # random subset for RLPD
        q_est = torch.min(
            *[
                critic(
                    experience_batch.observation_next,
                    self._prep_action(policy_output.action),
                    experience_batch.lang,
                ).q
                for critic in critic_targets
            ]
        )
        return q_est

    def _compute_q_estimates(self, experience_batch):
        return [
            critic(
                experience_batch.observation,
                self._prep_action(experience_batch.action.detach()),
                experience_batch.lang,
            ).q.squeeze(-1)
            for critic in self.critics
        ]  # no gradient propagation into policy here!
