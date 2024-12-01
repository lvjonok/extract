import os

from extract.models.skill_prior_mdl import ImageSkillPriorMdl
from extract.components.logger import Logger
from extract.utils.general_utils import AttrDict
from extract.configs.default_data_configs.libero import data_spec
from extract.components.evaluator import TopOfNSequenceEvaluator

current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    "model": ImageSkillPriorMdl,
    "logger": Logger,
    "data_dir": "./datasets/processed_libero_dataset_lowres",
    "evaluator": TopOfNSequenceEvaluator,
    "num_epochs": 76,
    "epoch_cycles_train": 50,
    "top_of_n_eval": 100,
    "top_comp_metric": "mse",
    "lr": 1e-3,
    "use_amp": False,
}
configuration = AttrDict(configuration)

lang_dim = 384

data_spec.res = 84

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_rollout_steps=10,
    nz_vae=10,
    kl_div_weight=1e-3,
    nz_enc=128,
    nz_mid=128,
    n_input_frames=2,
    prior_input_res=data_spec.res,
    encoder_max_range=2.0,
    use_language=True,
    lang_dim=lang_dim,
)

# set use_image to True
data_spec.use_image = True
# lang conditioned
data_spec.use_language = True
# lang dim
data_spec.lang_dim = lang_dim


# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = (
    model_config.n_rollout_steps + model_config.n_input_frames
)  # flat last action from seq gets cropped
data_config.dataset_spec.n_input_frames = model_config.n_input_frames
