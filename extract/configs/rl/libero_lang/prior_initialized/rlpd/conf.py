from extract.configs.rl.libero_lang.base_conf import *
from extract.configs.skill_prior_learning.libero_lang.flat.conf import (
    configuration as old_configuration,
)
from extract.configs.skill_prior_learning.libero_lang.flat.conf import (
    data_config as old_data_config,
)
from extract.rl.components.replay_buffer import IntImageUniformReplayBuffer
from extract.rl.policies.prior_policies import ACLanguagePriorInitializedPolicy
from extract.models.bc_mdl import ImageBCMdl

# update agent
agent_config.policy = ACLanguagePriorInitializedPolicy

# rlpd specific params

agent_old_data_config = AttrDict(data_dir=old_configuration.data_dir)
agent_old_data_config.update(old_data_config)
agent_config.update(
    AttrDict(
        num_critics=10,
        n_update_iterations=10,
        critic_subset=2,
        critic_mean_for_policy_loss=True,
        sample_old_data=True,
        old_data_config=agent_old_data_config,
        policy_update_every=10,  # default is to have it the same as the n_update_iterations (# critic updates)
        # n_warmup_steps=5e3,
    )
)
critic_params.update(
    AttrDict(
        norm="layer",
    )
)
configuration.agent = LangConditionedSACAgent
agent_config.replay = IntImageUniformReplayBuffer

policy_params.update(
    AttrDict(
        prior_model=ImageBCMdl,
        prior_model_params=AttrDict(
            state_dim=data_spec.state_dim,
            action_dim=data_spec.n_actions,
            input_res=data_spec.res,
            n_input_frames=sampler_config.n_frames,
            use_language=True,
            lang_dim=lang_dim,
        ),
        prior_model_checkpoint=os.path.join(
            os.environ["EXP_DIR"], "skill_prior_learning/block_stacking/flat"
        ),
    )
)
