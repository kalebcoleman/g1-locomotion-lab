"""PPO runner configs: IDENTICAL for both lanes (Isaac Lab's own G1 flat PPO settings), only the experiment name differs."""
from isaaclab.utils import configclass
from isaaclab_tasks.core.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg


@configclass
class RecPPOCfg(G1FlatPPORunnerCfg):
    lane = "x"
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = f"g1_recovery_{self.lane}"
        self.max_iterations = 1500
        self.save_interval = 50


@configclass
class RecPPOCfgA(RecPPOCfg):
    lane = "a"


@configclass
class RecPPOCfgB(RecPPOCfg):
    lane = "b"
