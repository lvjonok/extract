import os

from extract.models.cluster_skill_prior_mdl import ImageClusterSkillPriorMdl
from extract.data.libero.libero_data_loader import (
    SkillClusterLIBEROSequenceSplitDataset,
)
from extract.components.logger import Logger
from extract.utils.general_utils import AttrDict
from extract.configs.default_data_configs.libero import data_spec
from extract.components.evaluator import TopOfNSequenceEvaluator

current_dir = os.path.dirname(os.path.realpath(__file__))

data_spec.dataset_class = SkillClusterLIBEROSequenceSplitDataset
data_spec.cluster_data_file = "FILL IN"


configuration = {
    "model": ImageClusterSkillPriorMdl,
    "logger": Logger,
    "data_dir": "./datasets/processed_libero_dataset_lowres/",
    "num_epochs": 76,
    "evaluator": TopOfNSequenceEvaluator,
    "top_of_n_eval": 100,
    "epoch_cycles_train": 50,
    "top_comp_metric": "mse",
    "use_amp": False,
    "lr": 1e-3,
    "optimizer": "adamw",
}
configuration = AttrDict(configuration)


lang_dim = 384

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    kl_div_weight=1e-3,  # testing to see if restricting the encoder max range can help reproduce the same results
    nz_enc=128,
    nz_mid=128,
    encoder_max_range=2.0,
    n_skills=None,  # FILL IN
    n_input_frames=2,
    prior_input_res=data_spec.res,
    nz_vae=5,
    n_lstm_layers=1,
    use_language=True,
    lang_dim=lang_dim,
)
# set use_image to True
data_spec.use_image = True
# lang conditioned
data_spec.use_language = True
# lang dim
data_spec.lang_dim = lang_dim


# data_spec.pad_n_steps = (
#   model_config.n_input_frames - 1
# )  # pad extra frames at beginning of sequence to allow for n_input_frames to be used TODO: implement this for libero maybe, currently not working

# Dataset
data_config = AttrDict()
data_spec.n_input_frames = model_config.n_input_frames
data_config.dataset_spec = data_spec
