"""G1 recovery tasks (registered on import; import AFTER the simulation launcher, like Isaac Lab's own task packages)."""
import gymnasium as gym
from . import common  # noqa

gym.register(id="G1-Recovery-A-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": f"{__name__}.lane_a:G1RecoveryAEnvCfg",
                     "rsl_rl_cfg_entry_point": f"{__name__}.agents:RecPPOCfgA", "default_agent": "rsl_rl"})
gym.register(id="G1-Recovery-A-Assist-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": f"{__name__}.lane_a:G1RecoveryAAssistEnvCfg",
                     "rsl_rl_cfg_entry_point": f"{__name__}.agents:RecPPOCfgA", "default_agent": "rsl_rl"})
try:
    gym.register(id="G1-Recovery-B-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                 kwargs={"env_cfg_entry_point": f"{__name__}.lane_b:G1RecoveryBEnvCfg",
                         "rsl_rl_cfg_entry_point": f"{__name__}.agents:RecPPOCfgB", "default_agent": "rsl_rl"})
except Exception:
    pass


for _id, _cls in (("G1-Recovery-R1-v0", "G1RecoveryR1EnvCfg"), ("G1-Recovery-R1-Eval-v0", "G1RecoveryR1EvalEnvCfg"), ("G1-Recovery-R2-v0", "G1RecoveryR2EnvCfg"), ("G1-Recovery-R2-Eval-v0", "G1RecoveryR2EvalEnvCfg"), ("G1-Recovery-R3-v0", "G1RecoveryR3EnvCfg"), ("G1-Recovery-R3-Eval-v0", "G1RecoveryR3EvalEnvCfg"), ("G1-Recovery-R4-v0", "G1RecoveryR4EnvCfg"), ("G1-Recovery-R4n-v0", "G1RecoveryR4nEnvCfg"), ("G1-Recovery-R5-v0", "G1RecoveryR5EnvCfg"), ("G1-Recovery-R5-Eval-v0", "G1RecoveryR5EvalEnvCfg")):
    gym.register(id=_id, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                 kwargs={"env_cfg_entry_point": f"{__name__}.lane_a:{_cls}", "rsl_rl_cfg_entry_point": f"{__name__}.agents:RecPPOCfgR", "default_agent": "rsl_rl"})
