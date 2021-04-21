from gym.envs.registration import register

# ----------------------------------------
# mjrl environments
# ----------------------------------------


register(
    id='mjrl_essim-v0',
    entry_point='mjrl.envs:EssimEnv',
    max_episode_steps=500,
)

from mjrl.envs.mujoco_env import MujocoEnv

from mjrl.envs.essim import EssimEnv
