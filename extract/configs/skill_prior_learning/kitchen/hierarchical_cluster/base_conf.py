import os

from extract.models.cluster_skill_prior_mdl import ImageClusterSkillPriorMdl
from extract.data.kitchen.src.kitchen_data_loader import (
    SkillClusterD4RLSequenceSplitDataset,
)
from extract.components.logger import Logger
from extract.utils.general_utils import AttrDict
from extract.configs.default_data_configs.kitchen import data_spec
from extract.components.evaluator import TopOfNSequenceEvaluator

current_dir = os.path.dirname(os.path.realpath(__file__))

data_spec.dataset_class = SkillClusterD4RLSequenceSplitDataset
data_spec.cluster_data_file = "FILL IN"


configuration = {
    "model": ImageClusterSkillPriorMdl,
    "logger": Logger,
    "data_dir": "./datasets/kitchen_mixed_data.h5",
    "epoch_cycles_train": 10,
    "evaluator": TopOfNSequenceEvaluator,
    "top_of_n_eval": 100,
    "top_comp_metric": "mse",
    "use_amp": False,
    "lr": 1e-3,
    "optimizer": "adamw",
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    # n_rollout_steps=10,
    kl_div_weight=1e-3,  # testing to see if restricting the encoder max range can help reproduce the same results
    nz_enc=128,
    nz_mid=128,
    encoder_max_range=2.0,
    n_skills=None,  # FILL IN
    n_input_frames=4,
    prior_input_res=data_spec.res,
    n_lstm_layers=1,
    num_prior_net_layers=4,
    nz_vae=5,
)
# set use_image to True
data_spec.use_image = True
data_spec.pad_n_steps = (
    model_config.n_input_frames - 1
)  # pad extra frames at beginning of sequence to allow for n_input_frames to be used
data_spec.n_input_frames = model_config.n_input_frames

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.max_subseq_len = (
    # 30 + data_spec.pad_n_steps - 1
    30
    + model_config.n_input_frames
)  # no +1 like in spirl because we are defining max_subseq_len in terms of the action lengths
data_config.dataset_spec.min_subseq_len = 5 + model_config.n_input_frames
# data_config.dataset_spec.min_subseq_len = 5 + data_spec.pad_n_steps
