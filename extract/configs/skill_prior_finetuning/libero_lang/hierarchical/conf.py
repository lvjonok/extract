import os
from extract.configs.skill_prior_learning.libero_lang.hierarchical.conf import *
from extract.configs.hrl.libero_lang.spirl.conf import (
    agent_config,
    env_config,
    configuration,
    sampler_config,
)

current_dir = os.path.dirname(os.path.realpath(__file__))


training_configuration = {
    "model": ImageSkillPriorMdl,
    "logger": Logger,
    "data_dir": "./datasets/processed_libero_dataset_lowres",
    "evaluator": TopOfNSequenceEvaluator,
    "num_epochs": 300,
    "epoch_cycles_train": 50,
    "top_of_n_eval": 100,
    "top_comp_metric": "mse",
    "lr": 1e-3,
    "use_amp": False,
    "rollout": True,
}
training_configuration = AttrDict(training_configuration)
# update the rl configuration with these params
configuration.update(training_configuration)

model_config.freeze_encoder = False
model_config.freeze_decoder = False
model_config.ckpt_path = f"{os.environ['EXP_DIR']}/skill_prior_learning/libero_lang/hierarchical/"
data_config.finetune = True
data_config.finetune_dataset = "FILL IN"
