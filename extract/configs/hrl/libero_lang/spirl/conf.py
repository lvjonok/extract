# from extract.configs.hrl.kitchen_image.base_conf import *
from extract.rl.policies.prior_policies import ACLanguageLearnedPriorAugmentedPIPolicy
from extract.rl.agents.prior_sac_agent import LanguageConditionedActionPriorSACAgent
import os
import copy

from extract.utils.general_utils import AttrDict
from extract.rl.components.agent import FixedIntervalHierarchicalAgent
from extract.rl.envs.libero import LIBEROEnv
from extract.rl.components.sampler import MultiImageAugmentedHierarchicalSampler
from extract.rl.components.replay_buffer import LanguageIntImageUniformReplayBuffer
from extract.rl.components.critic import LangConditionedConvCritic
from extract.rl.agents.skill_space_agent import ACSkillSpaceAgent
from extract.models.skill_prior_mdl import ImageSkillPriorMdl
from extract.configs.default_data_configs.libero import data_spec

current_dir = os.path.dirname(os.path.realpath(__file__))

notes = "hierarchical RL on the kitchen env"

configuration = {
    "seed": 42,
    "agent": FixedIntervalHierarchicalAgent,
    "environment": LIBEROEnv,
    "sampler": MultiImageAugmentedHierarchicalSampler,
    "data_dir": ".",
    "num_epochs": 25,
    "max_rollout_len": 300,
    "n_steps_per_epoch": 50000,
    "n_warmup_steps": 1e3,
}
configuration = AttrDict(configuration)


# Replay Buffer
replay_params = AttrDict(capacity=1e5)

# Observation Normalization
obs_norm_params = AttrDict()

frame_stack = 2

sampler_config = AttrDict(
    n_frames=frame_stack,
)

base_agent_params = AttrDict(
    batch_size=256,
    replay=LanguageIntImageUniformReplayBuffer,
    replay_params=replay_params,
    # obs_normalizer=Normalizer,
    # obs_normalizer_params=obs_norm_params,
    discount_factor=0.99,
    clip_q_target=True,
)

data_spec.res = 84
###### Low-Level ######
# LL Policy
ll_model_params = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_rollout_steps=10,
    kl_div_weight=5e-4,
    nz_enc=128,
    nz_mid=128,
    nz_vae=10,
    n_processing_layers=5,
    n_input_frames=frame_stack,
    prior_input_res=data_spec.res,
    encoder_max_range=2.0,
    use_language=True,
    lang_dim=384,
)

# LL Agent
ll_agent_config = copy.deepcopy(base_agent_params)
ll_agent_config.update(
    AttrDict(
        model=ImageSkillPriorMdl,
        model_params=ll_model_params,
        model_checkpoint=os.path.join(
            os.environ["EXP_DIR"],
            "skill_prior_learning/libero/hierarchical/SPIRL_libero_pretran_res128_kl1e-3_0/",
            # "skill_prior_learning/libero/hierarchical/SPIRL_libero_prtran_kl1e-4_0/",
            # "skill_prior_learning/libero/hierarchical/SPIRL_libero_prtran_kl5e-4_0/",
        ),
    )
)


###### High-Level ########
# HL Policy
hl_policy_params = AttrDict(
    action_dim=ll_model_params.nz_vae,  # z-dimension of the skill VAE
    input_dim=data_spec.state_dim,
    max_action_range=2.0,  # prior is Gaussian with unit variance
    input_res=ll_model_params.prior_input_res,
    input_nc=3 * ll_model_params.n_input_frames,
    nz_mid=256,
    n_layers=5,
    prior_model=ll_agent_config.model,
    prior_model_params=ll_agent_config.model_params,
    prior_model_checkpoint=ll_agent_config.model_checkpoint,
    analytic_kl=False,  # false is the current default
)

# HL Critic
hl_critic_params = AttrDict(
    action_dim=hl_policy_params.action_dim,
    input_dim=hl_policy_params.input_dim,
    input_res=ll_model_params.prior_input_res,
    input_nc=3 * ll_model_params.n_input_frames,
    output_dim=1,
    n_layers=5,  # number of policy network layers
    nz_mid=256,
    lang_dim=ll_model_params.lang_dim,
    action_input=True,
)

# HL Agent
hl_agent_config = copy.deepcopy(base_agent_params)
hl_agent_config.update(
    AttrDict(
        # policy=ConvPolicy,
        policy=ACLanguageLearnedPriorAugmentedPIPolicy,
        policy_params=hl_policy_params,
        critic=LangConditionedConvCritic,
        critic_params=hl_critic_params,
        fixed_alpha=0.01,
        # td_schedule_params=AttrDict(p=5.0),
    )
)


##### Joint Agent #######
agent_config = AttrDict(
    hl_agent=LanguageConditionedActionPriorSACAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=ACSkillSpaceAgent,
    ll_agent_params=ll_agent_config,
    hl_interval=ll_model_params.n_rollout_steps,
    log_video_caption=True,
)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec

# Environment
env_config = AttrDict(
    task_id=None,
    task_suite=None,
    reward_norm=1.0,
    screen_height=ll_model_params.prior_input_res,
    screen_width=ll_model_params.prior_input_res,
    randomize_initial_state=False,
)

# reduce replay capacity because we are training image-based, do not dump (too large)
from extract.rl.components.replay_buffer import SplitObsUniformReplayBuffer

agent_config.ll_agent_params.replay = SplitObsUniformReplayBuffer
agent_config.ll_agent_params.replay_params.unused_obs_size = (
    ll_model_params.prior_input_res**2 * 3 * 2
    + hl_agent_config.policy_params.action_dim
)  # ignore HL action
agent_config.ll_agent_params.replay_params.dump_replay = False
agent_config.hl_agent_params.replay_params.dump_replay = False
