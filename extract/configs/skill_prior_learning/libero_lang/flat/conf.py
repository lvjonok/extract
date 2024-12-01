import os

from extract.models.bc_mdl import ImageBCMdl
from extract.components.logger import Logger
from extract.utils.general_utils import AttrDict
from extract.configs.default_data_configs.libero import data_spec
from extract.components.evaluator import DummyEvaluator


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    "model": ImageBCMdl,
    "logger": Logger,
    "data_dir": "./datasets/processed_libero_dataset_lowres",
    "epoch_cycles_train": 50,
    "num_epochs": 76,
    "evaluator": DummyEvaluator,
    "lr": 1e-3,
    "use_amp": False,
}
configuration = AttrDict(configuration)
data_spec.res = 84
lang_dim = 384
model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    input_res=data_spec.res,
    n_input_frames=2,
    use_language=True,
    lang_dim=lang_dim,
)

data_spec.use_image = True
data_spec.use_language = True
data_spec.lang_dim = lang_dim

data_spec.pad_n_steps = (
    model_config.n_input_frames - 1
)  # pad extra frames at beginning of sequence to allow for n_input_frames to be used
data_spec.n_input_frames = model_config.n_input_frames
# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = (
    1 + model_config.n_input_frames
)  # flat last action from seq gets cropped
