import os

from extract.utils.general_utils import AttrDict
from extract.rl.agents.ac_agent import SACAgent
from extract.rl.components.sampler import ACMultiImageAugmentedSampler
from extract.rl.policies.mlp_policies import ConvPolicy
from extract.rl.components.critic import ConvCritic
from extract.rl.components.replay_buffer import IntImageUniformReplayBuffer
from extract.rl.envs.kitchen import KitchenEnv
from extract.configs.default_data_configs.kitchen import data_spec


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = "non-hierarchical RL experiments in Kitchen Image env"

configuration = {
    "seed": 42,
    "agent": SACAgent,
    "environment": KitchenEnv,
    "sampler": ACMultiImageAugmentedSampler,
    "data_dir": ".",
    "num_epochs": 60,
    "max_rollout_len": 280,
    "n_steps_per_epoch": 50000,
    "n_warmup_steps": 5e3,
}
configuration = AttrDict(configuration)

sampler_config = AttrDict(
    n_frames=4,
)

# Policy
policy_params = AttrDict(
    input_nc=3 * sampler_config.n_frames,
    action_dim=data_spec.n_actions,
    input_res=data_spec.res,
    input_dim=data_spec.state_dim,
    n_layers=3,  #  reduce num layers for stability in these baselines
    nz_mid=256,
    max_action_range=1.0,
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
)

# Replay Buffer
replay_params = AttrDict(
    capacity=1e5,
    dump_replay=False,
)

# Agent
agent_config = AttrDict(
    policy=ConvPolicy,
    policy_params=policy_params,
    critic=ConvCritic,
    critic_params=critic_params,
    replay=IntImageUniformReplayBuffer,
    replay_params=replay_params,
    clip_q_target=True,
    batch_size=256,
    log_video_caption=True,
    policy_lr=1e-4,
    critic_lr=1e-4,
)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec

# Environment
env_config = AttrDict(
    reward_norm=1.0,
    screen_height=data_spec.res,
    screen_width=data_spec.res,
)
