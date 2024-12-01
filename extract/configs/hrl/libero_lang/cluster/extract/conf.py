from extract.configs.hrl.libero_lang.cluster.base_conf import *

ll_agent_config.model_params.n_skills = 8
ll_agent_config.model_params.max_rollout_steps = 40
ll_agent_config.model_checkpoint = os.path.join(
    os.environ["EXP_DIR"],
    "skill_prior_finetuning/libero_lang/hierarchical_cluster/extract/",
)

# update policy to use prior model for computing divergence
hl_policy_params.update(
    AttrDict(
        action_dim=ll_agent_config.model_params.nz_vae
        + ll_agent_config.model_params.n_skills,
        input_dim=data_spec.state_dim,
        prior_model=ll_agent_config.model,
        prior_model_params=ll_agent_config.model_params,
        prior_model_checkpoint=ll_agent_config.model_checkpoint,
    )
)

hl_critic_params.output_dim = ll_agent_config.model_params.n_skills

# update agent, set target divergence
agent_config.hl_agent_params.update(
    AttrDict(
        fixed_alpha=0.1,
        fixed_alpha_d=0.1,
    )
)

agent_config.ll_agent_params.replay_params.unused_obs_size = (
    ll_model_params.prior_input_res**2 * 3 * 2
    + hl_agent_config.policy_params.action_dim
)  # ignore HL action

configuration.num_epochs = 60
replay_params.capacity = 1e5
