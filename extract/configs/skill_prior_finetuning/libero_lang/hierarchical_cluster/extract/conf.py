from extract.configs.skill_prior_learning.libero_lang.hierarchical_cluster.base_conf import *
from extract.configs.hrl.libero_lang.cluster.extract.conf import agent_config, env_config, sampler_config
from extract.configs.hrl.libero_lang.cluster.extract.conf import configuration as rl_configuration
import os

data_spec.cluster_data_file = (
    "../generated_dataset_clusters/libero/r3m-99_first_diff_KMeans_n_clusters:8_MedFilt(7)"
)
model_config.n_skills = 8
data_config.dataset_spec.start_skill_anywhere = False

data_config.dataset_spec.max_subseq_len = (
    40 + model_config.n_input_frames
)  # no +1 like in spirl because we are defining max_subseq_len in terms of the action lengths
data_config.dataset_spec.min_subseq_len = 5 + model_config.n_input_frames


# finetuning specific
configuration.rollout = True

# update the configuration to be shared
rl_configuration.update(configuration)
configuration = rl_configuration
configuration.num_epochs = 300

model_config.freeze_encoder=False
model_config.freeze_decoder=False
model_config.ckpt_path = f"{os.environ['EXP_DIR']}/skill_prior_learning/libero_lang/hierarchical_cluster/extract/"
data_config.finetune = True
data_config.finetune_dataset = "FILL IN"
