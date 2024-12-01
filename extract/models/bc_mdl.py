from contextlib import contextmanager
import torch
import torch.nn as nn
import copy

from extract.components.base_model import BaseModel
from extract.modules.losses import NLL, L2Loss
from extract.modules.subnetworks import Predictor, Encoder
from extract.utils.general_utils import AttrDict, ParamDict
from extract.utils.pytorch_utils import RemoveSpatial, ResizeSpatial
from extract.modules.variational_inference import (
    ProbabilisticModel,
    MultivariateGaussian,
)
from extract.modules.layers import LayerBuilderParams
from extract.modules.mdn import GMM, MDN


class BCMdl(BaseModel):
    """Simple recurrent forward predictor network with image encoder and decoder."""

    def __init__(self, params, logger=None):
        BaseModel.__init__(self, logger)
        ProbabilisticModel.__init__(self)
        self._hp = self._default_hparams()
        self._hp.overwrite(params)  # override defaults with config file
        self._hp.builder = LayerBuilderParams(
            self._hp.use_convs, self._hp.normalization
        )
        self.device = self._hp.device

        self.build_network()

    def _default_hparams(self):
        # put new parameters in here:
        return (
            super()
            ._default_hparams()
            .overwrite(
                ParamDict(
                    {
                        "use_convs": False,
                        "device": None,
                        "state_dim": 1,  # dimensionality of the state space
                        "action_dim": 1,  # dimensionality of the action space
                        "nz_mid": 128,  # number of dimensions for internal feature spaces
                        "n_processing_layers": 5,  # number of layers in MLPs
                        "output_type": "gauss",  # distribution type for learned prior, ['gauss', 'gmm', 'flow']
                        "n_gmm_prior_components": 5,  # number of Gaussian components for GMM learned prior
                        "use_language": False,
                        "lang_dim": 384,
                    }
                )
            )
        )

    def build_network(self):
        assert not self._hp.use_convs  # currently only supports non-image inputs
        # assert self._hp.output_type == "gauss"  # currently only support unimodal output
        in_dim = self._hp.state_dim
        if self._hp.use_language:
            in_dim += self._hp.lang_dim
        if self._hp.output_type == "gmm":
            self.net = torch.nn.Sequential(
                Predictor(self._hp, input_size=in_dim, output_size=self._hp.nz_mid),
                MDN(
                    input_size=self._hp.nz_mid,
                    output_size=self._hp.action_dim,
                    num_gaussians=10,
                ),
            )
        else:
            self.net = Predictor(
                self._hp, input_size=in_dim, output_size=self._hp.action_dim * 2
            )

    def forward(self, inputs, use_learned_prior=False):
        """
        forward pass at training time
        """
        output = AttrDict()

        output.pred_act = self._compute_output_dist(self._net_inputs(inputs))
        return output

    def loss(self, model_output, inputs):
        losses = AttrDict()

        # reconstruction loss
        if self._hp.output_type == "gmm":
            losses.nll = NLL()(model_output.pred_act, self._regression_targets(inputs))
        else:
            losses.l2 = L2Loss()(
                model_output.pred_act.sample(), self._regression_targets(inputs)
            )
            # MH: For deterministic output, we don't need to sample.
            # losses.l2 = L2Loss()(
            #     model_output.pred_act.mu, self._regression_targets(inputs)
            # )

        losses.total = self._compute_total_loss(losses)
        return losses

    def _compute_output_dist(self, inputs):
        if self._hp.output_type == "gmm":
            return GMM(*self.net(inputs))
        elif self._hp.output_type == "flow":
            return self.net(inputs)
        else:
            return MultivariateGaussian(self.net(inputs))

    def _net_inputs(self, inputs):
        if self._hp.use_language:
            return torch.cat(
                [
                    inputs.states[:, 0],
                    inputs.lang[:, 0],
                ],
                dim=1,
            )
        return inputs.states[:, 0]

    def _regression_targets(self, inputs):
        return inputs.actions[:, 0]

    def compute_learned_prior(self, inputs, first_only=False):
        """Used in BC prior regularized RL policies."""
        assert first_only is True  # do not currently support ensembles for BC model
        if len(inputs.shape) == 1:
            return self._compute_output_dist(inputs[None])[0]
        else:
            return self._compute_output_dist(inputs)

    @property
    def resolution(self):
        return 64  # return dummy resolution, images are not used by this model

    @contextmanager
    def val_mode(self):
        pass
        yield
        pass


class ImageBCMdl(BCMdl):
    """Implements BC model with image input."""

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "input_res": 32,  # input resolution
                "encoder_ngf": 8,  # number of feature maps in shallowest level of encoder
                "n_input_frames": 1,  # number of input frames
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
                img_sz=self._hp.input_res,  # image resolution
                input_nc=3 * self._hp.n_input_frames,  # number of input feature maps
                ngf=self._hp.encoder_ngf,  # number of feature maps in shallowest level
                nz_enc=self._hp.nz_mid,  # size of image encoder output feature
                builder=LayerBuilderParams(
                    use_convs=True, normalization=self._hp.normalization
                ),
            )
        )

    def build_network(self):
        in_size = self._hp.nz_mid
        if self._hp.use_language:
            in_size += self._hp.lang_dim
        last_layer = Predictor(
            self._hp,
            input_size=in_size,
            output_size=self._hp.action_dim * 2,
        )
        if self._hp.use_language:
            self.net = LangConditionedPolicy(
                self._hp, self._updated_encoder_params(), last_layer
            )
        else:
            self.net = nn.Sequential(
                ResizeSpatial(self._hp.input_res),
                Encoder(self._updated_encoder_params()),
                RemoveSpatial(),
                last_layer,
            )

    def _net_inputs(self, inputs):
        images = inputs.images[:, : self._hp.n_input_frames].reshape(
            inputs.images.shape[0], -1, self.resolution, self.resolution
        )
        if self._hp.use_language:
            return AttrDict(
                images=images,
                lang=inputs.lang[:, 0],
                shape=images.shape,  # for backwards compat
            )
        else:
            return images

    def _regression_targets(self, inputs):
        return inputs.actions[:, (self._hp.n_input_frames - 1)]

    def unflatten_obs(self, raw_obs):
        """Utility to unflatten [obs, prior_obs] concatenated observation (for RL usage)."""
        # if len(raw_obs.shape) == 1:
        #    raw_obs = raw_obs[None]
        assert (
            len(raw_obs.shape) == 2
            and raw_obs.shape[1]
            == self._hp.state_dim
            + self._hp.input_res**2 * 3 * self._hp.n_input_frames
        )
        return AttrDict(
            obs=raw_obs[:, self._hp.state_dim],
            prior_obs=raw_obs[:, self._hp.state_dim :].reshape(
                raw_obs.shape[0],
                3 * self._hp.n_input_frames,
                self._hp.input_res,
                self._hp.input_res,
            ),
        )

    @property
    def resolution(self):
        return self._hp.input_res


class LangConditionedPolicy(nn.Module):
    """
    Combines the image and language
    """

    def __init__(self, hp, encoder_params, policy_net):
        super().__init__()
        self._image_processor = nn.Sequential(
            ResizeSpatial(hp.input_res),
            Encoder(encoder_params),
            RemoveSpatial(),
        )

        self._policy_net = policy_net

    def forward(self, inputs):
        images = inputs.images
        lang = inputs.lang
        if len(lang.shape) == 3:
            lang = lang[:, 0]
        image_out = self._image_processor(images)
        flattened_input = torch.cat((image_out, lang), dim=-1)
        return self._policy_net(flattened_input)
