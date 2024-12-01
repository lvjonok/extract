from extract.rl.policies.prior_policies import LearnedPriorAugmentedPIPolicy
from extract.utils.pytorch_utils import no_batchnorm_update
from extract.utils.general_utils import AttrDict

import torch


class HybridLearnedPriorAugmentedPIPolicy(LearnedPriorAugmentedPIPolicy):
    """Computes prior divergence for discrete and continuous latents."""

    def __init__(self, *args, **kwargs):
        LearnedPriorAugmentedPIPolicy.__init__(self, *args, **kwargs)

    def forward(self, obs, extra_info):
        """Computes action distribution & samples. First computes discrete output, then continuous argument."""
        # compute output distributions and actions
        with no_batchnorm_update(self.net):
            original_obs = obs
            discrete_output_dist = self.net.compute_learned_discrete_prior(original_obs)
            discrete_skill_choice = discrete_output_dist.sample().unsqueeze(1)
            #if self._is_train:
            #    discrete_skill_choice = discrete_output_dist.sample().unsqueeze(1)
            #else:
            #    discrete_skill_choice = torch.argmax(
            #        discrete_output_dist.logits, dim=-1
            #    ).unsqueeze(1)
            lang = None
            if "lang" in extra_info:
                lang = extra_info.lang
            if isinstance(obs, AttrDict):
                # from _prep_input
                obs = obs.images
            continuous_inputs = AttrDict(
                states=obs,
                images=obs,
                lang=lang,
                skills=discrete_skill_choice,
            )
            continuous_output_dist = self.net.compute_learned_continuous_prior(
                self.net._learned_continuous_prior_input(continuous_inputs)
            )
            continuous_output_action = continuous_output_dist.rsample()

            #if self._is_train:
            #    continuous_output_action = continuous_output_dist.rsample()
            #else:
            #    continuous_output_action = continuous_output_dist.mu

        # compute log prob
        log_prob_discrete = discrete_output_dist.log_prob(discrete_skill_choice)
        log_prob_continuous = continuous_output_dist.log_prob(continuous_output_action)

        # optionally squash continuous outputs
        if self._hp.squash_output_dist:
            continuous_output_action, log_prob_continuous = self._tanh_squash_output(
                continuous_output_action, log_prob_continuous
            )

        # compute (dummy) aggregate terms
        output_dist = continuous_output_dist  # this is a dummy output
        action = torch.cat((discrete_skill_choice, continuous_output_action), dim=-1)
        log_prob = (log_prob_discrete + log_prob_continuous) / 2

        policy_output = AttrDict(
            action=action,
            log_prob=log_prob,
            dist=output_dist,
            prob_d=discrete_output_dist.probs,
            action_d=discrete_skill_choice,
        )

        if not self._rollout_mode:
            # compute prior and posterior distributions
            with no_batchnorm_update(self.prior_net):
                with torch.no_grad():
                    discrete_prior_dist = self.prior_net.compute_learned_discrete_prior(
                        original_obs
                    )
                    continuous_prior_inputs = AttrDict(
                        states=obs,
                        images=obs,
                        lang=lang,
                        skills=discrete_skill_choice,
                    )
                    continuous_prior_dist = (
                        self.prior_net.compute_learned_continuous_prior(
                            self.prior_net._learned_continuous_prior_input(
                                continuous_prior_inputs
                            )
                        )
                    )

            # compute divergences
            policy_output.discrete_prior_divergence = self.clamp_divergence(
                discrete_output_dist.kl_divergence(discrete_prior_dist)
            ).mean(dim=-1)
            policy_output.continuous_prior_divergence = self.clamp_divergence(
                continuous_output_dist.kl_divergence(continuous_prior_dist)
            ).mean(dim=-1)

        return policy_output


class ACHybridLearnedPriorAugmentedPIPolicy(HybridLearnedPriorAugmentedPIPolicy):
    """HybridLearnedPriorAugmentedPIPolicy for case with separate prior obs --> uses prior observation as input only."""

    def forward(self, obs, extra_info):
        if obs.shape[0] == 1:
            return super().forward(
                self.net.unflatten_obs(obs).prior_obs, extra_info
            )  # use policy_net or batch_size 1 inputs
        return super().forward(self.prior_net.unflatten_obs(obs).prior_obs, extra_info)


class LanguageHybridLearnedPriorAugmentedPIPolicy(HybridLearnedPriorAugmentedPIPolicy):
    def _prep_inputs(self, obs, extra_info):
        # this follows skill_prior_mdl.py's def _learned_prior_input in the state based model
        assert "lang" in extra_info
        inputs = AttrDict(
            states=obs,
            lang=extra_info.lang.squeeze(1),
        )
        return inputs

    def forward(self, obs, extra_info):
        return super().forward(self._prep_inputs(obs, extra_info), extra_info)


class ACLanguageHybridLearnedPriorAugmentedPIPolicy(
    LanguageHybridLearnedPriorAugmentedPIPolicy
):
    """LanguageLearnedPriorAugmentedPIPolicy for case with separate prior obs --> uses prior observation as input only + language."""

    def __init__(self, config):
        super().__init__(config)  # this is fsr necessary for it not to throw an error

    def _prep_inputs(self, obs, extra_info):
        # this follows cluster_skill_prior_mdl.py's def _learned_prior_input in the Image Based model
        assert "lang" in extra_info
        return AttrDict(
            images=obs,
            lang=extra_info.lang.squeeze(1),
            shape=obs.shape,  # for backwards compat
        )

    def forward(self, obs, extra_info):
        if obs.shape[0] == 1:
            return super().forward(
                self.net.unflatten_obs(
                    self._prep_inputs(obs, extra_info).images
                ).prior_obs,
                extra_info,
            )  # use policy_net or batch_size 1 inputs
        return super().forward(
            self.prior_net.unflatten_obs(obs).prior_obs,
            extra_info,
        )
