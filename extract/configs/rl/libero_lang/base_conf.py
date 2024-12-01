import os

from extract.utils.general_utils import AttrDict
from extract.rl.agents.ac_agent import LangConditionedSACAgent
from extract.rl.components.sampler import ACMultiImageAugmentedSampler
from extract.rl.policies.mlp_policies import LangConditionedConvPolicy
from extract.rl.components.critic import LangConditionedConvCritic
from extract.rl.components.replay_buffer import LanguageIntImageUniformReplayBuffer
from extract.rl.envs.libero import LIBEROEnv
from extract.configs.default_data_configs.libero import data_spec


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = "non-hierarchical RL experiments in LIBERO env"

configuration = {
    "seed": 42,
    "agent": LangConditionedSACAgent,
    "environment": LIBEROEnv,
    "sampler": ACMultiImageAugmentedSampler,
    "data_dir": ".",
    "num_epochs": 25,
    "max_rollout_len": 300,
    "n_steps_per_epoch": 50000,
    "n_warmup_steps": 5e3,
}
configuration = AttrDict(configuration)

sampler_config = AttrDict(
    n_frames=2,
)

lang_dim = 384

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
    # norm="layer",  # for stability
    lang_dim=lang_dim,
)

# Replay Buffer
replay_params = AttrDict(
    capacity=1e5,
    dump_replay=False,
)

# Agent
agent_config = AttrDict(
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

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec

# Environment
env_config = AttrDict(
    task_id=None,
    task_suite="libero_10",
    reward_norm=1.0,
    screen_height=policy_params.input_res,
    screen_width=policy_params.input_res,
    randomize_initial_state=False,
)
