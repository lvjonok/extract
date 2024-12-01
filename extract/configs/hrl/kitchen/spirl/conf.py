from extract.configs.hrl.kitchen_image.base_conf import *
from extract.rl.policies.prior_policies import ACLearnedPriorAugmentedPIPolicy
from extract.rl.agents.prior_sac_agent import ActionPriorSACAgent


# update policy to use prior model for computing divergence
hl_policy_params.update(
    AttrDict(
        prior_model=ll_agent_config.model,
        prior_model_params=ll_agent_config.model_params,
        prior_model_checkpoint=ll_agent_config.model_checkpoint,
        analytic_kl=False,  # false is the current default
    )
)
hl_agent_config.policy = ACLearnedPriorAugmentedPIPolicy

# update agent, set target divergence
agent_config.hl_agent = ActionPriorSACAgent
agent_config.hl_agent_params.update(
    AttrDict(
        td_schedule_params=AttrDict(p=5.0),
        fixed_alpha=None,
    )
)
