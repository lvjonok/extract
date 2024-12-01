import torch
import torch.nn as nn
import copy

from extract.utils.general_utils import ParamDict, AttrDict
from extract.modules.layers import LayerBuilderParams
from extract.modules.subnetworks import Encoder, Predictor, HybridConvMLPEncoder


class Critic(nn.Module):
    """Base critic class."""

    def __init__(self):
        super().__init__()
        self._net = self._build_network()

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "action_dim": 1,  # dimensionality of the action space
                "normalization": "none",  # normalization used in critic network ['none', 'batch', 'group', 'layer']
                "action_input": True,  # forward takes actions as second argument if set to True
            }
        )
        return default_dict

    def forward(
        self, obs, actions: torch.Tensor = None, extra_conditioning: torch.Tensor = None
    ):
        raise NotImplementedError("Needs to be implemented by child class.")

    @staticmethod
    def dummy_output():
        return AttrDict(q=None)

    def _build_network(self):
        """Constructs the policy network."""
        raise NotImplementedError("Needs to be implemented by child class.")


class MLPCritic(Critic):
    """MLP-based critic."""

    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self._hp.builder = LayerBuilderParams(
            use_convs=False, normalization=self._hp.normalization
        )
        super().__init__()

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "input_dim": 32,  # dimensionality of the observation input
                "n_layers": 3,  # number of policy network layers
                "nz_mid": 64,  # size of the intermediate network layers
                "output_dim": 1,  # number of outputs, can be >1 for discrete action spaces
            }
        )
        return super()._default_hparams().overwrite(default_dict)

    def forward(
        self, obs, actions: torch.tensor = None, extra_conditioning: torch.tensor = None
    ):
        input = torch.cat((obs, actions), dim=-1) if self._hp.action_input else obs
        if extra_conditioning is not None:
            input = torch.cat((input, extra_conditioning), dim=-1)
        return AttrDict(q=self._net(input))

    def _build_network(self):
        input_size = (
            self._hp.input_dim + self._hp.action_dim
            if self._hp.action_input
            else self._hp.input_dim
        )
        return Predictor(
            self._hp,
            input_size=input_size,
            output_size=self._hp.output_dim,
            mid_size=self._hp.nz_mid,
            num_layers=self._hp.n_layers,
            spatial=False,
        )


class ConvCritic(MLPCritic):
    """Critic that can incorporate image and action inputs by fusing conv and MLP encoder."""

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "input_res": 32,  # resolution of the image input
                "input_nc": 3,  # number of input channels
                "ngf": 8,  # number of channels in shallowest layer of image encoder
                "nz_enc": 64,  # number of dimensions in encoder-latent space
            }
        )
        return super()._default_hparams().overwrite(default_dict)

    def _build_network(self):
        return HybridConvMLPEncoder(
            copy.deepcopy(self._hp).overwrite(AttrDict(input_dim=self._hp.action_dim))
        )

    def forward(self, obs, actions, extra_conditioning: torch.tensor = None):
        split_obs = AttrDict(
            vector=actions,
            image=obs[:, self._hp.input_dim :].reshape(
                obs.shape[0], -1, self._hp.input_res, self._hp.input_res
            ),
        )
        return AttrDict(q=self._net(split_obs))


class LangConditionedConvCritic(ConvCritic):
    """
    Combines the image and language
    """

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "lang_dim": 384,  # dimensionality of the language input
            }
        )
        return super()._default_hparams().overwrite(default_dict)

    def _build_network(self):
        return HybridConvMLPEncoder(
            copy.deepcopy(self._hp).overwrite(
                AttrDict(input_dim=self._hp.action_dim + self._hp.lang_dim)
            )
        )

    def forward(self, obs, actions, extra_conditioning: torch.tensor):
        # extra_conditioning should be the language embedding
        if len(extra_conditioning.shape) == 4:
            extra_conditioning = extra_conditioning.squeeze(1)
        if len(extra_conditioning.shape) == 3:
            extra_conditioning = extra_conditioning.squeeze(1)
        split_obs = AttrDict(
            vector=torch.cat((actions, extra_conditioning), dim=-1),
            image=obs[:, self._hp.input_dim :].reshape(
                obs.shape[0], -1, self._hp.input_res, self._hp.input_res
            ),
        )
        return AttrDict(q=self._net(split_obs))


class SplitObsMLPCritic(MLPCritic):
    """Splits off unused part of observations."""

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "unused_obs_size": None,  # dimensionality of split off observation part
                "discard_part": "back",  # which part of observation to discard ['front', 'back']
            }
        )
        return super()._default_hparams().overwrite(default_dict)

    def forward(self, raw_obs, *args, **kwargs):
        if self._hp.discard_part == "front":
            return super().forward(
                raw_obs[:, self._hp.unused_obs_size :], *args, **kwargs
            )
        elif self._hp.discard_part == "back":
            return super().forward(
                raw_obs[:, : -self._hp.unused_obs_size], *args, **kwargs
            )
        else:
            raise ValueError(
                "Cannot parse discard_part parameter {}!".format(self._hp.discard_part)
            )
