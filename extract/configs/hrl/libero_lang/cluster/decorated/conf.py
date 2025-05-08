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

configuration.num_epochs = 20
replay_params.capacity = 1e5

# prior that it is the same configuration as extract
from extract.rl.components.agent import ResidualAgent
from extract.rl.agents.ac_agent import SACAgent
from extract.rl.policies.mlp_policies import ConvPolicy
from extract.rl.components.critic import ConvCritic

# we update configuration to use ResidualAgent
base_agent_cls = configuration.agent
configuration.update({"agent": ResidualAgent})

# TODO: right now they are defaulting, need to set the params
residual_agent_params = copy.deepcopy(base_agent_params)
# hl_critic_params.output_dim = data_spec.n_actions
# hl_critic_params.action_dim = data_spec.n_actions
# hl_policy_params.output_dim = data_spec.n_actions
# hl_policy_params.action_dim = data_spec.n_actions
# agent_config.ll_agent_params.output_dim = data_spec.n_actions
# residual_agent_params.update(
#     AttrDict(
#         policy=LangConditionedConvPolicy,
#         policy_params=hl_policy_params,
#         critic=LangConditionedConvCritic,
#         critic_params=hl_critic_params,
#         fixed_alpha=None,
#         fixed_alpha_d=None,
#         discount_factor=0.99,
#     )
# )

sampler_config = AttrDict(
    n_frames=2,
)

lang_dim = 384

# # Policy
# policy_params = AttrDict(
#     input_nc=3 * sampler_config.n_frames,
#     action_dim=data_spec.n_actions,
#     input_res=data_spec.res,
#     input_dim=data_spec.state_dim,
#     n_layers=3,  #  reduce num layers for stability in these baselines
#     nz_mid=256,
#     max_action_range=1.0,
#     lang_dim=lang_dim,
# )

# # Critic
# critic_params = AttrDict(
#     action_dim=policy_params.action_dim,
#     input_dim=policy_params.input_dim,
#     input_res=policy_params.input_res,
#     input_nc=3 * sampler_config.n_frames,
#     output_dim=1,
#     n_layers=policy_params.n_layers,
#     nz_mid=256,
#     action_input=True,
#     # norm="layer",  # for stability
#     lang_dim=lang_dim,
# )

# residual_agent_params.update(
#     AttrDict(
#         policy=ConvPolicy,
#         policy_params=policy_params,
#         critic=ConvCritic,
#         critic_params=critic_params,
#         fixed_alpha=None,
#         fixed_alpha_d=None,
#         discount_factor=0.99,
#         # Trying to fix issue with replay buffer and language
#         replay=LanguageIntImageUniformReplayBuffer,  # save memory by saving as np.uint8
#         replay_params=replay_params,
#         model_params=AttrDict(
#             use_language=True,
#             lang_dim=lang_dim,
#         ),
#     )
# )

from extract.utils.general_utils import AttrDict
from extract.rl.agents.ac_agent import LangConditionedSACAgent
from extract.rl.components.sampler import ACMultiImageAugmentedSampler
from extract.rl.policies.mlp_policies import LangConditionedConvPolicy
from extract.rl.components.critic import LangConditionedConvCritic
from extract.rl.components.replay_buffer import LanguageIntImageUniformReplayBuffer
from extract.rl.envs.libero import LIBEROEnv

# Policy
policy_params = AttrDict(
    input_nc=3 * sampler_config.n_frames,
    action_dim=data_spec.n_actions,
    input_res=data_spec.res,
    input_dim=data_spec.state_dim,
    n_layers=3,  #  reduce num layers for stability in these baselines
    nz_mid=256,
    max_action_range=1.0,
    lang_dim=lang_dim,
)

# Critic
critic_params = AttrDict(
    action_dim=policy_params.action_dim,
    input_dim=policy_params.input_dim,
    input_res=policy_params.input_res,
    input_nc=3 * sampler_config.n_frames,
    output_dim=1,
    n_layers=policy_params.n_layers,
    nz_mid=256,
    action_input=True,
    norm="layer",  # for stability
    lang_dim=lang_dim,
)

# Replay Buffer
replay_params = AttrDict(
    capacity=1e5,
    dump_replay=False,
)

# Agent
residual_agent_config = AttrDict(
    policy=LangConditionedConvPolicy,
    policy_params=policy_params,
    critic=LangConditionedConvCritic,
    critic_params=critic_params,
    replay=LanguageIntImageUniformReplayBuffer,
    replay_params=replay_params,
    clip_q_target=True,
    batch_size=256,
    log_video_caption=True,
    policy_lr=1e-4,
    critic_lr=1e-4,
    discount_factor=0.99,
)

configuration.sampler = ACMultiImageAugmentedSampler

agent_config = AttrDict(
    base_agent=base_agent_cls,
    base_agent_params=agent_config,
    residual_agent=LangConditionedSACAgent,
    residual_agent_params=residual_agent_config,
)
