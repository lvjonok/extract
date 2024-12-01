import os

from extract.configs.skill_prior_learning.libero_lang.flat.conf import *
from extract.configs.rl.libero_lang.prior_initialized.bc_finetune.conf import (
    agent_config,
    env_config,
    configuration,
    sampler_config,
)

current_dir = os.path.dirname(os.path.realpath(__file__))
training_configuration = {
    "model": ImageBCMdl,
    "logger": Logger,
    "data_dir": "./datasets/processed_libero_dataset_lowres",
    "evaluator": DummyEvaluator,
    "num_epochs": 100,
    "epoch_cycles_train": 50,
    "lr": 1e-3,
    "use_amp": False,
    "rollout": True,
}
training_configuration = AttrDict(training_configuration)
# update the rl configuration with these params
configuration.update(training_configuration)

model_config.ckpt_path = f"{os.environ['EXP_DIR']}/skill_prior_learning/libero_lang/flat/"
data_config.finetune = True
data_config.finetune_dataset = "FILL IN"
