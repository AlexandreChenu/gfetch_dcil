import os
import gym
from gym.envs.registration import register

__version__ = "0.1.0"


def envpath():
    resdir = os.path.join(os.path.dirname(__file__))
    return resdir


print("gym-gfetch: ")
print("|    gym version and path:", gym.__version__, gym.__path__)

print("|    REGISTERING GFetch-v0 from", envpath())
register(
    id="GFetch-v0",
    entry_point="gym_gfetch.envs:GFetch",
    max_episode_steps=200,
    reward_threshold=1.,
)

print("|    REGISTERING GFetchGoal-v0 from", envpath())
register(
    id="GFetchGoal-v0",
    entry_point="gym_gfetch.envs:GFetchGoal",
    max_episode_steps=200,
    reward_threshold=1.,
)

print("|    REGISTERING GFetchDCIL-v0 from", envpath())
register(
    id="GFetchDCIL-v0",
    entry_point="gym_gfetch.envs:GFetchDCIL",
    reward_threshold=1.,
)
