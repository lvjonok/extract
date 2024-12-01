from extract.configs.skill_prior_learning.libero_lang.hierarchical_cluster.base_conf import *

data_spec.cluster_data_file = (
    "../generated_dataset_clusters/libero/r3m-99_first_diff_KMeans_n_clusters:8_MedFilt(7)"
)
model_config.n_skills = 8
data_config.dataset_spec.start_skill_anywhere = False

data_config.dataset_spec.max_subseq_len = (
    40 + model_config.n_input_frames
)  # no +1 like in spirl because we are defining max_subseq_len in terms of the action lengths
data_config.dataset_spec.min_subseq_len = 5 + model_config.n_input_frames
