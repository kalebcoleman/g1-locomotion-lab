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
