from extract.configs.skill_prior_learning.kitchen.hierarchical_cluster.base_conf import *

data_spec.cluster_data_file = "../generated_dataset_clusters/kitchen/r3m-99_diff_KMeans_n_clusters:8_MedFilt(7)/mixed/clusters.h5"
model_config.n_skills = 8
data_config.dataset_spec.start_skill_anywhere = False
