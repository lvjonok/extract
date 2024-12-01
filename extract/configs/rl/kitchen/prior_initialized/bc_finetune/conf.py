from extract.configs.rl.kitchen.base_conf import *
from extract.rl.policies.prior_policies import ACPriorInitializedPolicy
from extract.models.bc_mdl import ImageBCMdl

# update agent
agent_config.policy = ACPriorInitializedPolicy
configuration.agent = SACAgent

policy_params.update(
    AttrDict(
        prior_model=ImageBCMdl,
        prior_model_params=AttrDict(
            state_dim=data_spec.state_dim,
            action_dim=data_spec.n_actions,
            input_res=data_spec.res,
            n_input_frames=sampler_config.n_frames,
        ),
        prior_model_checkpoint=os.path.join(
            os.environ["EXP_DIR"], "skill_prior_learning/kitchen/flat"
        ),
    )
)
