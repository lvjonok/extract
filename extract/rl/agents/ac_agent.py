import torch
import os
import numpy as np

from extract.rl.components.agent import BaseAgent
from extract.utils.general_utils import ParamDict, map_dict, AttrDict
from extract.utils.pytorch_utils import (
    ten2ar,
    avg_grad_norm,
    TensorModule,
    check_shape,
    map2torch,
    map2np,
)
from extract.rl.utils.mpi import sync_networks


class ACAgent(BaseAgent):
    """Implements actor-critic agent. (does not implement update function, this should be handled by RL algo agent)"""

    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self._hp = self._default_hparams().overwrite(config)
        self.policy = self._hp.policy(self._hp.policy_params)
        if self.policy.has_trainable_params:
            self.policy_opt = self._get_optimizer(
                self._hp.optimizer, self.policy, self._hp.policy_lr
            )

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "policy": None,  # policy class
                "policy_params": None,  # parameters for the policy class
                "policy_lr": 3e-4,  # learning rate for policy update
            }
        )
        return super()._default_hparams().overwrite(default_dict)

    def _act(self, obs, extra_info=None):
        # TODO implement non-sampling validation mode
        obs = map2torch(self._obs_normalizer(obs), self._hp.device)
        if extra_info is not None:
            extra_info = map2torch(extra_info, self._hp.device)
        if len(obs.shape) == 1:  # we need batched inputs for policy
            policy_output = self._remove_batch(self.policy(obs[None], extra_info))
            if "dist" in policy_output:
                del policy_output["dist"]
            return map2np(policy_output)
        return map2np(self.policy(obs, extra_info))

    def _act_rand(self, obs, extra_info=None):
        if extra_info is not None:
            extra_info = map2torch(extra_info, self._hp.device)
        policy_output = self.policy.sample_rand(
            map2torch(obs, self.policy.device), extra_info
        )
        if "dist" in policy_output:
            del policy_output["dist"]
        return map2np(policy_output)

    def state_dict(self, *args, **kwargs):
        d = super().state_dict()
        if self.policy.has_trainable_params:
            d["policy_opt"] = self.policy_opt.state_dict()
        return d

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.policy_opt.load_state_dict(state_dict.pop("policy_opt"))
        super().load_state_dict(state_dict, *args, **kwargs)

    def visualize(self, logger, rollout_storage, step):
        super().visualize(logger, rollout_storage, step)
        self.policy.visualize(logger, rollout_storage, step)

    def reset(self):
        self.policy.reset()

    def sync_networks(self):
        if self.policy.has_trainable_params:
            sync_networks(self.policy)

    def _preprocess_experience(self, experience_batch):
        """Optionally pre-process experience before it is used for policy training."""
        return experience_batch


class SACAgent(ACAgent):
    """Implements SAC algorithm."""

    def __init__(self, config):
        ACAgent.__init__(self, config)
        self._hp = self._default_hparams().overwrite(config)
        self._num_critics = self._hp.num_critics
        self._num_critic_subset = self._hp.critic_subset

        # build critics and target networks, copy weights of critics to target networks
        self.critics = torch.nn.ModuleList(
            [self._hp.critic(self._hp.critic_params) for _ in range(self._num_critics)]
        )
        self.critic_targets = torch.nn.ModuleList(
            [self._hp.critic(self._hp.critic_params) for _ in range(self._num_critics)]
        )
        [
            self._copy_to_target_network(target, source)
            for target, source in zip(self.critics, self.critic_targets)
        ]

        # build optimizers for critics
        self.critic_opts = [
            self._get_optimizer(self._hp.optimizer, critic, self._hp.critic_lr)
            for critic in self.critics
        ]

        # define entropy multiplier alpha
        if self._hp.fixed_alpha is not None:
            self._log_alpha = TensorModule(
                np.log(self._hp.fixed_alpha)
                * torch.ones(1, requires_grad=False, device=self._hp.device)
            )
        else:
            self._log_alpha = TensorModule(
                torch.zeros(1, requires_grad=True, device=self._hp.device)
            )
            self.alpha_opt = self._get_optimizer(
                self._hp.optimizer, self._log_alpha, self._hp.alpha_lr
            )
        self._target_entropy = (
            self._hp.target_entropy
            if self._hp.target_entropy is not None
            else -1 * self._hp.policy_params.action_dim
        )

        # build replay buffer
        self.replay_buffer = self._hp.replay(self._hp.replay_params)

        if self._hp.sample_old_data:
            self.old_data_buffer = self.get_old_dataloader(
                config,
                self.policy,
                self._hp.old_data_config,
                "train",
                n_repeat=10000000,  # arbitrarily high number
            )

        self._update_steps = (
            0  # counts the number of alpha updates for optional variable schedules
        )

    def get_old_dataloader(
        self, args, model, data_conf, phase, n_repeat, dataset_size=-1
    ):
        assert (
            self._hp.batch_size % 2 == 0
        ), "BATCH SIZE MUST BE EVEN for 50/50 RLPD sampling"
        dataset_class = data_conf.dataset_spec.dataset_class
        # todo: fix the get_data_loader
        loader = dataset_class(
            data_conf.data_dir,
            data_conf,
            resolution=self.resolution,
            phase=phase,
            shuffle=True,
            dataset_size=dataset_size,
        ).get_data_loader(int(self._hp.batch_size // 2), n_repeat)

        return loader

    def sample_old_data(self):
        old_data_batch = next(self.old_data_buffer)
        # this will be a batch_size x dim tensor
        list_of_relevant_keys = [
            # "observation",
            "states",
            # "observation_next",
            "next_states",
            # "ll_action",
            "actions",
            "reward",
            "done",
        ]
        assert all([key in old_data_batch for key in list_of_relevant_keys])
        rl_old_data_batch = AttrDict(
            observation=old_data_batch["states"][:, 0],
            observation_next=old_data_batch["next_states"][:, 0],
            action=old_data_batch["actions"],
            reward=old_data_batch["reward"],
            done=old_data_batch["done"],
        )
        return rl_old_data_batch

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "critic": None,  # critic class
                "critic_params": None,  # parameters for the critic class
                "replay": None,  # replay buffer class
                "replay_params": None,  # parameters for replay buffer
                "critic_lr": 3e-4,  # learning rate for critic update
                "alpha_lr": 3e-4,  # learning rate for alpha coefficient update
                "fixed_alpha": None,  # optionally fixed value for alpha
                "reward_scale": 1.0,  # SAC reward scale
                "clip_q_target": False,  # if True, clips Q target
                "target_entropy": None,  # target value for automatic entropy tuning, if None uses -action_dim
                "num_critics": 2,  # rlpd argument, defaults to 2 for SAC
                "critic_subset": 2,  # rlpd argument, defaults to 2 for SAC
                "critic_mean_for_policy_loss": False,  # rlpd does mean over all critics for policy loss
                "sample_old_data": False,  # rlpd for sampling old data at a 50/50 ratio
                "old_data_config": None,  # rlpd for old data config
                "policy_update_every": 1,  # update policy every n critic iterations
            }
        )
        return super()._default_hparams().overwrite(default_dict)

    def _compute_critic_values(self, experience_batch, policy_output):
        """Computes values for all critics."""
        operation = torch.mean if self._hp.critic_mean_for_policy_loss else torch.min
        q_est = operation(
            *[
                critic(
                    experience_batch.observation,
                    self._prep_action(policy_output.action),
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
                ).q
                for critic in critic_targets
            ]
        )
        return q_est

    def update(self, experience_batch):
        """Updates actor and critics."""
        # push experience batch into replay buffer
        self.add_experience(experience_batch)

        for i in range(self._hp.update_iterations):
            # sample batch and normalize
            experience_batch = self._sample_experience()
            experience_batch = self._normalize_batch(experience_batch)
            experience_batch = map2torch(experience_batch, self._hp.device)
            experience_batch = self._preprocess_experience(experience_batch)

            # RLPD
            if self._hp.sample_old_data:
                old_experience_batch = self.sample_old_data(self.old_data_buffer)
                for key in experience_batch:
                    experience_batch[key] = torch.cat(
                        [experience_batch[key], old_experience_batch[key]], dim=0
                    )

            extra_info = AttrDict()
            if "lang" in experience_batch:
                extra_info["lang"] = experience_batch["lang"]

            policy_output = self._run_policy(experience_batch.observation, extra_info)

            # update alpha
            alpha_loss = self._update_alpha(experience_batch, policy_output)

            # compute policy loss
            if i % self._hp.policy_update_every == 0:
                policy_loss = self._compute_policy_loss(experience_batch, policy_output)

            # compute target Q value
            with torch.no_grad():
                policy_output_next = self._run_policy(
                    experience_batch.observation_next, extra_info
                )
                value_next = self._compute_next_value(
                    experience_batch, policy_output_next
                )
                q_target = (
                    experience_batch.reward * self._hp.reward_scale
                    + (1 - experience_batch.done)
                    * self._hp.discount_factor
                    * value_next
                )
                if self._hp.clip_q_target:
                    q_target = self._clip_q_target(q_target)
                q_target = q_target.detach()
                check_shape(q_target, [self._hp.batch_size])

            # compute critic loss
            critic_losses, qs = self._compute_critic_loss(
                experience_batch, q_target, policy_output
            )

            if (i + 1) % self._hp.policy_update_every == 0:
                # update policy network on policy loss
                self._perform_update(policy_loss, self.policy_opt, self.policy)

            # update critic networks
            [
                self._perform_update(critic_loss, critic_opt, critic)
                for critic_loss, critic_opt, critic in zip(
                    critic_losses, self.critic_opts, self.critics
                )
            ]

            # update target networks
            [
                self._soft_update_target_network(critic_target, critic)
                for critic_target, critic in zip(self.critic_targets, self.critics)
            ]

            # logging
            info = AttrDict(  # losses
                policy_loss=policy_loss,
                alpha_loss=alpha_loss,
                critic_loss_1=critic_losses[0],
                critic_loss_2=critic_losses[1],
            )
            if self._update_steps % 30 == 0:
                info.update(
                    AttrDict(  # gradient norms
                        policy_grad_norm=avg_grad_norm(self.policy),
                        critic_1_grad_norm=avg_grad_norm(self.critics[0]),
                        critic_2_grad_norm=avg_grad_norm(self.critics[1]),
                    )
                )
            info.update(
                AttrDict(  # misc
                    alpha=self.alpha,
                    pi_log_prob=policy_output.log_prob.mean(),
                    policy_entropy=policy_output.dist.entropy().mean(),
                    q_target=q_target.mean(),
                    q_1=qs[0].mean(),
                    q_2=qs[1].mean(),
                )
            )
            info.update(self._aux_info(experience_batch, policy_output))
            info = map_dict(ten2ar, info)

            self._update_steps += 1

        return info

    def add_experience(self, experience_batch):
        """Adds experience to replay buffer."""
        if not experience_batch:
            return  # pass if experience_batch is empty
        self.replay_buffer.append(experience_batch)
        self._obs_normalizer.update(experience_batch.observation)

    def _sample_experience(self):
        return self.replay_buffer.sample(n_samples=self._hp.batch_size)

    def _normalize_batch(self, experience_batch):
        """Optionally apply observation normalization."""
        experience_batch.observation = self._obs_normalizer(
            experience_batch.observation
        )
        experience_batch.observation_next = self._obs_normalizer(
            experience_batch.observation_next
        )
        return experience_batch

    def _run_policy(self, obs, extra_info):
        """Allows child classes to post-process policy outputs."""
        return self.policy(obs, extra_info)

    def _update_alpha(self, experience_batch, policy_output):
        if self._hp.fixed_alpha is not None:
            return 0.0
        alpha_loss = self._compute_alpha_loss(policy_output)
        self._perform_update(alpha_loss, self.alpha_opt, self._log_alpha)
        return alpha_loss

    def _compute_alpha_loss(self, policy_output):
        return (
            -1
            * (
                self.alpha * (self._target_entropy + policy_output.log_prob).detach()
            ).mean()
        )

    def _compute_policy_loss(self, experience_batch, policy_output):
        q_est = self._compute_critic_values(experience_batch, policy_output)
        policy_loss = -1 * q_est + self.alpha * policy_output.log_prob[:, None]
        check_shape(policy_loss, [self._hp.batch_size, 1])
        return policy_loss.mean()

    def _compute_next_value(self, experience_batch, policy_output):
        q_next = self._compute_next_critic_values(experience_batch, policy_output)
        next_val = q_next - self.alpha * policy_output.log_prob[:, None]
        check_shape(next_val, [self._hp.batch_size, 1])
        return next_val.squeeze(-1)

    def _compute_critic_loss(self, experience_batch, q_target, policy_output):
        qs = self._compute_q_estimates(experience_batch)
        check_shape(qs[0], [self._hp.batch_size])
        critic_losses = [0.5 * (q - q_target).pow(2).mean() for q in qs]
        return critic_losses, qs

    def _compute_q_estimates(self, experience_batch):
        return [
            critic(
                experience_batch.observation,
                self._prep_action(experience_batch.action.detach()),
            ).q.squeeze(-1)
            for critic in self.critics
        ]  # no gradient propagation into policy here!

    def _prep_action(self, action):
        """Preprocessing of action in case of discrete action space."""
        if len(action.shape) == 1:
            action = action[:, None]  # unsqueeze for single-dim action spaces
        return action.float()

    def _clip_q_target(self, q_target):
        clip = 1 / (1 - self._hp.discount_factor)
        return torch.clamp(q_target, -clip, clip)

    def _aux_info(self, experience_batch, policy_output):
        return AttrDict()

    def sync_networks(self):
        super().sync_networks()
        [sync_networks(critic) for critic in self.critics]
        sync_networks(self._log_alpha)

    def state_dict(self, *args, **kwargs):
        d = super().state_dict()
        d["critic_opts"] = [o.state_dict() for o in self.critic_opts]
        if self._hp.fixed_alpha is None:
            d["alpha_opt"] = self.alpha_opt.state_dict()
        return d

    def load_state_dict(self, state_dict, *args, **kwargs):
        [
            o.load_state_dict(d)
            for o, d in zip(self.critic_opts, state_dict.pop("critic_opts"))
        ]
        if self._hp.fixed_alpha is None:
            self.alpha_opt.load_state_dict(state_dict.pop("alpha_opt"))
        super().load_state_dict(state_dict, *args, **kwargs)

    def save_state(self, save_dir):
        """Saves compressed replay buffer to disk."""
        self.replay_buffer.save(os.path.join(save_dir, "replay"))

    def load_state(self, save_dir):
        """Loads replay buffer from disk."""
        self.replay_buffer.load(os.path.join(save_dir, "replay"))

    @property
    def alpha(self):
        return self._log_alpha().exp()

    @property
    def schedule_steps(self):
        return self._update_steps


class LangConditionedSACAgent(SACAgent):
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
        if self._num_critics == self._num_critic_subset:
            critic_targets = self.critic_targets
        else:
            critic_targets = self.critic_targets[
                np.random.choice(
                    self._num_critics, self._num_critic_subset, replace=False
                )
            ]
            # random subset for RLPD
        """Computes values for all critics."""
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
