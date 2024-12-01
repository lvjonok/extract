from contextlib import contextmanager
import copy
import torch
import torch.nn as nn
from collections import deque

from extract.utils.pytorch_utils import make_one_hot
from extract.models.skill_prior_mdl import (
    LangConditionedPrior,
    SkillPriorMdl,
    ImageSkillPriorMdl,
)
from extract.modules.losses import KLDivLoss, NLL, CELoss, L2Loss
from extract.modules.subnetworks import BaseProcessingLSTM, Predictor, Encoder
from extract.modules.recurrent_modules import RecurrentPredictor
from extract.utils.general_utils import (
    AttrDict,
    ParamDict,
    split_along_axis,
)
from extract.utils.pytorch_utils import (
    map2np,
    RemoveSpatial,
    ResizeSpatial,
    map2torch,
    find_tensor,
)
from extract.modules.variational_inference import (
    Gaussian,
    MultivariateGaussian,
    Categorical,
    get_fixed_prior,
)
from extract.modules.layers import LayerBuilderParams


class ClusterSkillPriorMdl(SkillPriorMdl):
    """Multi-Skill embedding + prior model for SPIRL algorithm."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._arange = torch.arange(128)

    def _default_hparams(self):
        default_dict = ParamDict()
        default_dict.update(
            {
                "n_skills": -1,  # number of discrete clusters variables
                "skill_progress_mse_weight": 1.0,
                "max_rollout_steps": 100,
                "skill_progress_termination_threshold": 0.99,
            }
        )

        # add new params to parent params
        parent_params = super()._default_hparams()
        parent_params.overwrite(default_dict)
        return parent_params

    def build_network(self):
        """Defines the network architecture (encoder aka inference net, decoder, prior)."""
        assert not self._hp.use_convs  # currently only supports non-image inputs
        self.q = self._build_inference_net()
        decoder_input_size = (
            self._hp.action_dim + self._hp.nz_vae + self._hp.n_skills + 1
        )  # + 1 for progress prediction
        if self._hp.use_language:
            decoder_input_size += self._hp.lang_dim
        self.decoder = RecurrentPredictor(
            self._hp,
            input_size=decoder_input_size,
            output_size=self._hp.action_dim + 1,  # + 1 for progress prediction
        )
        self.decoder_input_initalizer = self._build_decoder_initializer(
            size=self._hp.action_dim + 1  # + 1 for progress prediction
        )
        self.decoder_hidden_initalizer = self._build_decoder_initializer(
            size=self.decoder.cell.get_state_size()
        )

        self.p_z = self._build_continuous_skill_prior_net()
        self.p_k = self._build_discrete_skill_prior_net()

    def _build_continuous_skill_prior_net(self):
        """Builds Gaussian skill prior."""
        return Predictor(
            self._hp,
            input_size=self.continuous_prior_input_size,
            output_size=self._hp.nz_vae * 2,
            num_layers=self._hp.num_prior_net_layers,
            mid_size=self._hp.nz_mid_prior,
        )

    def _build_discrete_skill_prior_net(self):
        return Predictor(
            self._hp,
            input_size=self.discrete_prior_input_size,
            output_size=self._hp.n_skills,
            num_layers=self._hp.num_prior_net_layers,
            mid_size=self._hp.nz_mid_prior,
        )

    def forward(self, inputs, use_learned_prior=False):
        """Forward pass of the SPIRL model.
        :arg inputs: dict with 'states', 'actions', 'images' keys from data loader
        :arg use_learned_prior: if True, decodes samples from learned prior instead of posterior, used for RL
        """
        output = AttrDict()
        inputs.observations = inputs.actions  # for seamless evaluation

        # run inference
        output.q = self._run_inference(inputs)

        # compute (fixed) prior
        output.p = get_fixed_prior(output.q)

        # infer learned skill prior
        output.q_hat = self.compute_learned_continuous_prior(
            self._learned_continuous_prior_input(inputs)
        )
        output.d_hat = self.compute_learned_discrete_prior(
            self._learned_discrete_prior_input(inputs)
        )
        if use_learned_prior:
            output.p = output.q_hat  # use output of learned skill prior for sampling
            inputs.skills = output.d_hat.sample()  # sample discrete skill

        # sample latent variable
        output.z = output.p.sample() if self._sample_prior else output.q.sample()
        output.z_q = (
            output.z.clone() if not self._sample_prior else output.q.sample()
        )  # for loss computation

        # decode
        decoder_output = self.decode(
            output.z,
            steps=self._regression_targets(inputs).shape[1],
            inputs=inputs,
        )
        output.reconstruction = decoder_output[:, :, :-1]
        output.skill_progress = decoder_output[:, :, -1]
        return output

    # def _learned_prior_input(self, inputs):
    #    return torch.cat((inputs.states[:, 0], self.toOneHot(inputs.skills[:, 0])), dim=-1)

    def loss(self, model_output, inputs):
        """Loss computation of the SPIRL model.
        :arg model_output: output of SPIRL model forward pass
        :arg inputs: dict with 'states', 'actions', 'images' keys from data loader
        """
        losses = AttrDict()
        # reconstruction loss, assume unit variance model output Gaussian
        # keeping this NLL to be able to compare to prior spirl runs, but it really doesn't need to be NLL
        losses.rec_mse = NLL(self._hp.reconstruction_mse_weight)(
            Gaussian(
                model_output.reconstruction,
                torch.zeros_like(model_output.reconstruction),
            ),
            self._regression_targets(inputs),
            weights=self._action_pad_mask(inputs).unsqueeze(-1),
        )
        # skill progress prediction loss
        losses.skill_progress_mse = L2Loss(self._hp.skill_progress_mse_weight)(
            model_output.skill_progress,
            self._skill_progress_targets(inputs),
            weights=self._action_pad_mask(inputs),
        )

        # KL loss
        losses.kl_loss_z = KLDivLoss(self.beta)(model_output.q, model_output.p)

        # CE loss for discrete skill prior net
        losses.discrete_skill_ce = CELoss()(
            model_output.d_hat.logits,
            inputs.skills[:, 0].long(),
        )

        # learned skill prior net loss
        if self._hp.nll_prior_train:
            losses.q_hat_loss = NLL()(model_output.q_hat, model_output.z_q.detach())
        else:
            losses.q_hat_loss = KLDivLoss()(model_output.q.detach(), model_output.q_hat)

        # Optionally update beta
        if self.training and self._hp.target_kl is not None:
            self._update_beta(losses.kl_loss.value)

        losses.total = self._compute_total_loss(losses)
        return losses

    def _log_outputs(
        self,
        model_output,
        inputs,
        losses,
        step,
        log_images,
        phase,
        logger,
        **logging_kwargs
    ):
        """Optionally visualizes outputs of SPIRL model.
        :arg model_output: output of SPIRL model forward pass
        :arg inputs: dict with 'states', 'actions', 'images' keys from data loader
        :arg losses: output of SPIRL model loss() function
        :arg step: current training iteration
        :arg log_images: if True, log image visualizations (otherwise only scalar losses etc get logged automatically)
        :arg phase: 'train' or 'val'
        :arg logger: logger class, visualization functions should be implemented in this class
        """
        self._logger.log_scalar(self.beta, "beta", step, phase)

        # log videos/gifs in tensorboard
        if log_images:
            print("{} {}: logging videos".format(phase, step))
            self._logger.visualize(
                model_output, inputs, losses, step, phase, logger, **logging_kwargs
            )

    def decode(self, z, steps, inputs=None):
        """Runs forward pass of decoder given skill embedding.
        :arg z: skill embedding
        :arg cond_inputs: info that decoder is conditioned on
        :arg steps: number of steps decoder is rolled out
        """
        lstm_init_input = self.decoder_input_initalizer(inputs)
        lstm_init_hidden = self.decoder_hidden_initalizer(inputs)
        if "d_pred" in inputs:  # for training
            skill_input = inputs.d_pred
        else:
            skill_input = self.toOneHot(inputs.skills[:, 0])
        combined_static_input = torch.cat((z, skill_input), dim=-1)
        if self._hp.use_language:
            combined_static_input = torch.cat(
                (combined_static_input, inputs.lang[:, 0]), dim=-1
            )
        return self.decoder(
            lstm_initial_inputs=AttrDict(x_t=lstm_init_input),
            lstm_static_inputs=AttrDict(z=combined_static_input),
            steps=steps,
            lstm_hidden_init=lstm_init_hidden,
        ).pred

    def run(self, inputs, use_learned_prior=True):
        """Policy interface for model. Runs decoder if action plan is empty, otherwise returns next action from action plan.
        :arg inputs: dict with 'states', 'actions', 'images' keys from environment
        :arg use_learned_prior: if True, uses learned prior otherwise samples latent from uniform prior
        """
        inputs = map2torch(inputs, device=self.device)

        actions_and_progress = self.decode(
            inputs.z, steps=self._hp.max_rollout_steps, inputs=inputs
        )[0]
        actions, skill_progress = (
            actions_and_progress[:, :-1],
            actions_and_progress[:, -1],
        )
        threshold_mask = skill_progress >= self._hp.skill_progress_termination_threshold
        # get first instance in which skill progress exceeds threshold
        if threshold_mask.sum() > 0:
            last_action_index = torch.argmax(threshold_mask.float())
        else:
            last_action_index = self._hp.max_rollout_steps - 1

        action_plan = deque(
            split_along_axis(map2np(actions), axis=0)[: last_action_index + 1]
        )

        return action_plan

    def reset(self):
        """Resets action plan (should be called at beginning of episode when used in RL loop)."""
        self._action_plan = (
            deque()
        )  # stores action plan of LL policy when model is used as policy

    def _build_inference_net(self):
        # inference gets conditioned on state if decoding is also conditioned on state
        input_size = self._hp.action_dim + self._hp.n_skills
        if self._hp.use_language:
            input_size += self._hp.lang_dim
        return torch.nn.Sequential(
            BaseProcessingLSTM(self._hp, in_dim=input_size, out_dim=self._hp.nz_enc),
            torch.nn.Linear(self._hp.nz_enc, self._hp.nz_vae * 2),
        )

    def _build_decoder_initializer(self, size):
        class FixedTrainableInitializer(nn.Module):
            def __init__(self, hp):
                super().__init__()
                self._hp = hp
                self.val = nn.Parameter(
                    torch.zeros((1, size), requires_grad=True, device=self._hp.device)
                )

            def forward(self, state):
                return self.val.repeat(find_tensor(state).shape[0], 1)

        return FixedTrainableInitializer(self._hp)

    def _last_action_index(self, inputs):
        # last_action_offset = (
        #    0 if "n_input_frames" not in self._hp else self._hp.n_input_frames - 1
        # )
        # return inputs.last_action_index - last_action_offset
        if self._hp.cond_decode:
            return inputs.last_action_index - (self._hp.n_input_frames - 1)
        return inputs.last_action_index

    def _run_inference(self, inputs):
        skill_input = (
            self.toOneHot(inputs.skills[:, 0])
            .unsqueeze(1)
            .repeat(1, self._regression_targets(inputs).shape[1], 1)
        )
        inf_input = torch.cat((self._regression_targets(inputs), skill_input), dim=-1)
        if self._hp.use_language:
            inf_input = torch.cat((inf_input, inputs.lang), dim=-1)
        return MultivariateGaussian(
            self.q(inf_input)[self._arange, self._last_action_index(inputs)],
            max_mu=self._hp.encoder_max_range,
        )

    def compute_learned_continuous_prior(self, inputs):
        return MultivariateGaussian(self.p_z(inputs), max_mu=self._hp.encoder_max_range)

    def compute_learned_discrete_prior(self, inputs):
        return Categorical(logits=self.p_k(inputs))

    def toOneHot(self, sample):
        return make_one_hot(sample.long(), length=self._hp.n_skills).float()

    def _learned_continuous_prior_input(self, inputs):
        if "d_pred" in inputs:  # for learnable clustering training
            skill_input = inputs.d_pred
        else:
            skill_input = self.toOneHot(inputs.skills[:, 0])  # for RL
        if self._hp.use_language:
            skill_input = torch.cat((skill_input, inputs.lang), dim=-1)

        if len(inputs.states.shape) == 2:
            states = inputs.states
        else:
            states = inputs.states[:, 0]
        return torch.cat((states, skill_input), dim=1)

    def _learned_discrete_prior_input(self, inputs):
        if len(inputs.states.shape) == 2:  # for RL
            states = inputs.states
        else:
            states = inputs.states[:, 0]
        if self._hp.use_language:
            return torch.cat((states, inputs.lang), dim=1)
        return states

    def _regression_targets(self, inputs):
        if self._hp.cond_decode:
            return inputs.actions[:, (self._hp.n_input_frames - 1) :]
        return inputs.actions

    def _action_pad_mask(self, inputs):
        if self._hp.cond_decode:
            return inputs.action_pad_mask[:, (self._hp.n_input_frames - 1) :]
        return inputs.action_pad_mask

    def _skill_progress_targets(self, inputs):
        if self._hp.cond_decode:
            return inputs.skill_progress[:, (self._hp.n_input_frames - 1) :]
        return inputs.skill_progress

    @property
    def resolution(self):
        return 64  # return dummy resolution, images are not used by this model

    @property
    def latent_dim(self):
        return (
            self._hp.nz_vae + 1
        )  # + 1 for hl policy output having 1 extra dim for skill dimension

    @property
    def state_dim(self):
        return self._hp.state_dim

    @property
    def discrete_prior_input_size(self):
        if self._hp.use_language:
            return self.state_dim + self._hp.lang_dim
        return self.state_dim

    @property
    def continuous_prior_input_size(self):
        if self._hp.use_language:
            return self.state_dim + self._hp.n_skills + self._hp.lang_dim
        return self.state_dim + self._hp.n_skills

    @property
    def n_rollout_steps(self):
        raise NotImplementedError(
            "n_rollout_steps not implemented for this model because it has variable length skills"
        )

    @property
    def beta(self):
        return (
            self._log_beta().exp()[0].detach()
            if self._hp.target_kl is not None
            else self._hp.kl_div_weight
        )

    @property
    def max_rollout_steps(self):
        return self._hp.max_rollout_steps


class ImageClusterSkillPriorMdl(ClusterSkillPriorMdl, ImageSkillPriorMdl):
    """Implements learned skill prior with image input."""

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "prior_input_res": 64,  # input resolution of prior images
                "encoder_ngf": 8,  # number of feature maps in shallowest level of encoder
                "n_input_frames": 4,  # number of prior input frames
            }
        )
        # add new params to parent params
        return super()._default_hparams().overwrite(default_dict)

    def _updated_encoder_params(self):
        params = copy.deepcopy(self._hp)
        return params.overwrite(
            AttrDict(
                use_convs=True,
                use_skips=False,  # no skip connections needed flat we are not reconstructing
                img_sz=self._hp.prior_input_res,  # image resolution
                input_nc=3 * self._hp.n_input_frames,  # number of input feature maps
                ngf=self._hp.encoder_ngf,  # number of feature maps in shallowest level
                nz_enc=self._hp.nz_mid,  # size of image encoder output feature
                builder=LayerBuilderParams(
                    use_convs=True, normalization=self._hp.normalization
                ),
            )
        )

    def _build_discrete_skill_prior_net(self):
        if self._hp.use_language:
            return LangConditionedPrior(
                self._hp,
                self._updated_encoder_params(),
                super()._build_discrete_skill_prior_net(),
            )
        return nn.Sequential(
            ResizeSpatial(self._hp.prior_input_res),
            Encoder(self._updated_encoder_params()),
            RemoveSpatial(),
            super()._build_discrete_skill_prior_net(),
        )

    def _build_continuous_skill_prior_net(self):
        """Builds Gaussian skill prior."""
        return ImageAndSkillConditionedPrior(
            self._hp,
            self._updated_encoder_params(),
            super()._build_continuous_skill_prior_net(),
        )

    def _learned_discrete_prior_input(self, inputs):
        if len(inputs.images.shape) != 4:
            images = inputs.images[:, : self._hp.n_input_frames].reshape(
                inputs.images.shape[0], -1, self.resolution, self.resolution
            )
        else:
            images = inputs.images
        if self._hp.use_language:
            return AttrDict(
                images=images,
                lang=inputs.lang[:, 0],
                shape=images.shape,  # for backwards compat
            )
        else:
            return images

    @property
    def discrete_prior_input_size(self):
        if self._hp.use_language:
            return self._hp.nz_mid + self._hp.lang_dim
        return self._hp.nz_mid

    @property
    def continuous_prior_input_size(self):
        if self._hp.use_language:
            return self._hp.nz_mid + self._hp.n_skills + self._hp.lang_dim
        return self._hp.nz_mid + self._hp.n_skills

    @property
    def resolution(self):
        return self._hp.prior_input_res

    def _learned_continuous_prior_input(self, inputs):
        if "d_pred" in inputs:
            skill_input = inputs.d_pred
        else:
            skill_input = self.toOneHot(inputs.skills[:, 0])
        if len(inputs.images.shape) != 4:
            images = inputs.images[:, : self._hp.n_input_frames].reshape(
                inputs.images.shape[0], -1, self.resolution, self.resolution
            )
        else:
            images = inputs.images

        lang = None
        if self._hp.use_language:
            lang = inputs.lang[:, 0]
        return AttrDict(images=images, skill_input=skill_input, lang=lang)


class ImageAndSkillConditionedPrior(nn.Module):
    def __init__(self, hp, updated_encoder_params, state_continuous_prior_net):
        super().__init__()
        self._hp = hp
        self.conv_encoder = nn.Sequential(
            ResizeSpatial(self._hp.prior_input_res),
            Encoder(updated_encoder_params),
            RemoveSpatial(),
        )

        self.linear_encoder = state_continuous_prior_net

    def forward(self, input):
        mid_input = self.conv_encoder(input.images)
        if self._hp.use_language:
            lang = input.lang
            if len(lang.shape) == 3:
                lang = lang[:, 0]
            return self.linear_encoder(
                torch.cat((mid_input, input.skill_input, input.lang), dim=-1)
            )
        return self.linear_encoder(torch.cat((mid_input, input.skill_input), dim=-1))
