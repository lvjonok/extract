import os

from extract.models.bc_mdl import ImageBCMdl
from extract.components.logger import Logger
from extract.utils.general_utils import AttrDict
from extract.configs.default_data_configs.kitchen import data_spec
from extract.components.evaluator import DummyEvaluator


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    "model": ImageBCMdl,
    "logger": Logger,
    "data_dir": "./datasets/kitchen_mixed_data.h5",
    "epoch_cycles_train": 10,
    "evaluator": DummyEvaluator,
    "use_amp": False,
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    input_res=data_spec.res,
    n_input_frames=4,
)

data_spec.use_image = True
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
