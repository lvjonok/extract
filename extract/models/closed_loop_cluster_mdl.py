from contextlib import contextmanager
from collections import deque
from extract.utils.pytorch_utils import ResizeSpatial, RemoveSpatial, map2np, map2torch
import itertools
import copy
import torch
import torch.nn as nn
import numpy as np
from extract.models.cluster_skill_prior_mdl import (
    ClusterSkillPriorMdl,
    ImageClusterSkillPriorMdl,
)
from extract.modules.subnetworks import BaseProcessingLSTM, Predictor, Encoder
from extract.utils.general_utils import (
    batch_apply,
    split_along_axis,
)
from extract.modules.variational_inference import (
    MultivariateGaussian,
)


class ClClusterSkillPriorMdl(ClusterSkillPriorMdl):
    """Closed Loop Multi-Skill embedding + prior model for SPIRL algorithm."""

    def build_network(self):
        """Defines the network architecture (encoder aka inference net, decoder, prior)."""
        assert not self._hp.use_convs  # currently only supports non-image inputs
        assert (
            self._hp.cond_decode
        )  # need to decode based on state for closed-loop low-level
        self.q = self._build_inference_net()
        decoder_input_size = self.enc_size + self._hp.nz_vae + self._hp.n_skills
        if self._hp.use_language:
            decoder_input_size += self._hp.lang_dim
        self.decoder = Predictor(
            self._hp,
            input_size=decoder_input_size,
            output_size=self._hp.action_dim + 1,  # + 1 for progress prediction
            mid_size=self._hp.nz_mid_prior,
        )

        self.p_z = self._build_continuous_skill_prior_net()
        self.p_k = self._build_discrete_skill_prior_net()

    def run(self, inputs, use_learned_prior=True):
        """Policy interface for model. Runs decoder if action plan is empty, otherwise returns next action from action plan.
        :arg inputs: dict with 'states', 'actions', 'images' keys from environment
        :arg use_learned_prior: if True, uses learned prior otherwise samples latent from uniform prior
        """
        inputs = map2torch(inputs, device=self.device)
        decode_inputs = torch.cat(
            (inputs.states, inputs.z, self.toOneHot(inputs.skills[:, 0])), dim=-1
        )
        if self._hp.use_language:
            decode_inputs = torch.cat(
                (
                    decode_inputs,
                    inputs.lang[:, 0:1].repeat(1, inputs.states.shape[1], 1),
                ),
                dim=-1,
            )
        actions_and_progress = self.decoder(decode_inputs)
        actions, skill_progress = (
            actions_and_progress[:, :-1],
            actions_and_progress[:, -1],
        )
        if (skill_progress >= self._hp.skill_progress_termination_threshold).item():
            last_action = True
        else:
            last_action = False
        action_plan = deque(split_along_axis(map2np(actions), axis=0))

        return action_plan, last_action

    def decode(self, z, steps, inputs=None):
        """Runs forward pass of decoder given skill embedding.
        :arg z: skill embedding
        :arg cond_inputs: info that decoder is conditioned on
        :arg steps: number of steps decoder is rolled out
        """
        assert (
            inputs is not None
        )  # need additional state sequence input for full decode
        if "d_pred" in inputs:  # for training
            skill_input = inputs.d_pred
        else:
            skill_input = self.toOneHot(inputs.skills[:, 0])
        seq_enc = self._get_seq_enc(inputs)
        lang_input = None
        if self._hp.use_language:
            lang_input = inputs.lang[:, 0:1].repeat(1, steps, 1)
        return self._decode(
            seq_enc[:, :steps],
            z[:, None].repeat(1, steps, 1),
            skill_input[:, None].repeat(1, steps, 1),
            lang_input,
        )

    def _decode(self, input_states, z, skills, lang_input=None):
        decode_inputs = torch.cat((input_states, z, skills), dim=-1)
        if lang_input is not None:
            decode_inputs = torch.cat((decode_inputs, lang_input), dim=-1)
        return batch_apply(decode_inputs, self.decoder)

    def _build_inference_net(self):
        # inference gets conditioned on state if decoding is also conditioned on state
        input_size = self._hp.action_dim + self._hp.n_skills + self.prior_input_size
        return torch.nn.Sequential(
            BaseProcessingLSTM(self._hp, in_dim=input_size, out_dim=self._hp.nz_enc),
            torch.nn.Linear(self._hp.nz_enc, self._hp.nz_vae * 2),
        )

    def _get_seq_enc(self, inputs):
        return inputs.states[:, :-1]

    def enc_obs(self, obs):
        """Optionally encode observation for decoder."""
        return obs

    def _run_inference(self, inputs):
        # additionally add state sequence conditioning
        skill_input = (
            self.toOneHot(inputs.skills[:, 0])
            .unsqueeze(1)
            .repeat(1, self._regression_targets(inputs).shape[1], 1)
        )
        inf_input = torch.cat(
            (self._regression_targets(inputs), skill_input, self._get_seq_enc(inputs)),
            dim=-1,
        )
        return MultivariateGaussian(
            self.q(inf_input)[self._arange, self._last_action_index(inputs)],
            max_mu=self._hp.encoder_max_range,
        )

    @property
    def enc_size(self):
        return self._hp.state_dim


class ImageClClusterSPiRLMdl(ClClusterSkillPriorMdl, ImageClusterSkillPriorMdl):
    """Implements learned skill prior with image input."""

    def _default_hparams(self):
        return ImageClusterSkillPriorMdl._default_hparams(self)

    def _updated_encoder_params(self):
        return ImageClusterSkillPriorMdl._updated_encoder_params(self)

    def _learned_discrete_prior_input(self, inputs):
        return ImageClusterSkillPriorMdl._learned_discrete_prior_input(self, inputs)

    def _build_prior_net(self):
        return ImageClusterSkillPriorMdl._build_prior_net(self)

    def _build_inference_net(self):
        self.img_encoder = nn.Sequential(
            ResizeSpatial(self._hp.prior_input_res),  # encodes image inputs
            Encoder(self._updated_encoder_params()),
            RemoveSpatial(),
        )
        return ClClusterSkillPriorMdl._build_inference_net(self)

    def _learned_continuous_prior_input(self, inputs):
        return ImageClusterSkillPriorMdl._learned_continuous_prior_input(self, inputs)

    def enc_obs(self, obs):
        """Optionally encode observation for decoder."""
        return self.img_encoder(obs)

    @property
    def discrete_prior_input_size(self):
        return self._hp.nz_mid

    @property
    def continuous_prior_input_size(self):
        return self._hp.nz_mid + self._hp.n_skills

    def _get_seq_enc(self, inputs):
        # stack input image sequence
        stacked_imgs = torch.cat(
            [
                inputs.images[:, t : t + self._regression_targets(inputs).shape[1]]
                for t in range(self._hp.n_input_frames)
            ],
            dim=2,
        )
        # encode stacked seq
        return batch_apply(stacked_imgs, self.img_encoder)

    @property
    def enc_size(self):
        return self._hp.nz_mid

    @property
    def prior_input_size(self):
        if self._hp.use_language:
            return self.enc_size + self._hp.lang_dim
        return self.enc_size
