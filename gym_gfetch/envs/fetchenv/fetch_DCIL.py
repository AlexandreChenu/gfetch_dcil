
# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.

# See the License for the specific language governing permissions and
# limitations under the License.


# from .import_ai import *
from import_ai import *

from abc import ABC
from abc import abstractmethod
from typing import Optional

from fetch_env import ComplexFetchEnv

from typing import Union
from gym import utils, spaces
from gym import error
import numpy as np
import torch
from matplotlib import collections as mc
# from IPython import embed

from skill_manager_fetchenv import SkillsManager

import gym
gym._gym_disable_underscore_compat = True

import types
os.environ["PATH"] = os.environ["PATH"].replace('/usr/local/nvidia/bin', '')
try:
    import mujoco_py
    import gym.envs.robotics.utils
    import gym.envs.robotics.rotations
except Exception:
    print('WARNING: could not import mujoco_py. This means robotics environments will not work')
import gym.spaces
from scipy.spatial.transform import Rotation
from collections import defaultdict, namedtuple
import os
from gym.envs.mujoco import mujoco_env

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")


class GoalEnv(gym.Env):
    """The GoalEnv class that was migrated from gym (v0.22) to gym-robotics"""

    def reset(self, options=None, seed: Optional[int] = None, infos=None):
        super().reset(seed=seed)
        # Enforce that each GoalEnv uses a Goal-compatible observation space.
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise error.Error(
                "GoalEnv requires an observation space of type gym.spaces.Dict"
            )
        for key in ["observation", "achieved_goal", "desired_goal"]:
            if key not in self.observation_space.spaces:
                raise error.Error('GoalEnv requires the "{}" key.'.format(key))

    @abstractmethod
    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the step reward.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to
            the desired goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'],
                                                    ob['desired_goal'], info)
        """
        raise NotImplementedError


@torch.no_grad()
def default_reward_fun(action, new_obs):
    return 0.

## interface vers ComplexFetchEnv
class GFetch(gym.Env, utils.EzPickle, ABC):
    TARGET_SHAPE = 0
    MAX_PIX_VALUE = 0

    def __init__(self, num_envs=1, model_file='teleOp_boxes_1.xml', nsubsteps=20, min_grip_score=0, max_grip_score=0,
                 target_single_shelf=False, combine_table_shelf_box=False, ordered_grip=False,
                 target_location='1000', timestep=0.002, force_closed_doors=False):


        self.device = device
        self.num_envs = num_envs
        self.envs = [ComplexFetchEnv(
            model_file=model_file, nsubsteps=nsubsteps,
            min_grip_score=min_grip_score, max_grip_score=max_grip_score,
            #ret_full_state=False,
            ret_full_state=True,
            target_single_shelf=target_single_shelf,
            combine_table_shelf_box=combine_table_shelf_box, ordered_grip=ordered_grip,
            target_location=target_location, timestep=timestep,
            force_closed_doors=force_closed_doors
        ) for i in range(self.num_envs)]
        self.env = self.envs[0]

        self.single_action_space = self.env.action_space
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space,
            self.num_envs)

        self.single_observation_space = self.env.observation_space
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space,
            self.num_envs)

        self.set_reward_function(default_reward_fun)

        self.init_state = self.env.reset()
        self.init_state = torch.tensor(
            np.tile(np.array(self.init_state), (self.num_envs, 1))
        ).to(self.device)

        print("self.init_state.shape = ", self.init_state.shape)
        print("type(self.ini_state) = ", type(self.init_state))
        self.init_sim_state = self.env.sim.get_state()


        self.state = torch.clone(self.init_state)
        self.done = torch.zeros((self.num_envs, 1)).int().to(self.device)
        self.steps = torch.zeros((self.num_envs,1), dtype=torch.int).to(self.device)

        self.max_episode_steps = 10

        self.rooms = []

    def __getattr__(self, e):
        assert self.env is not self
        return getattr(self.env, e)

    def set_reward_function(self, reward_function):
        self.reward_function = (
            reward_function  # the reward function is not defined by the environment
        )

    def reset_model(self, env_indices=None) -> np.ndarray:
        """
        Reset environments to initial simulation state & return vector state
        """
        if env_indices is not None: ## reset selected environments only
            for env_indx in env_indices:
                self.envs[env_indx[0]].sim.set_state(self.init_sim_state)

        else: ### reset every environment
            for env in self.envs:
                env.sim.set_state(self.init_sim_state)

        return self.state_vector()

    def reset(self, options=None, seed: Optional[int] = None, infos=None):
        self.reset_model()
        self.steps = torch.zeros((self.num_envs,1), dtype=torch.int).to(self.device)
        return self.state_vector()

    def reset_done(self, options=None, seed: Optional[int] = None, infos=None):

        to_reset = list(torch.clone(self.done).flatten().nonzero().numpy())

        self.reset_model(env_indices = to_reset)

        zeros = torch.zeros((self.num_envs,1), dtype=torch.int).to(self.device)
        self.steps = torch.where(self.done==1, zeros, self.steps)

        return self.state_vector()

    def step(self, action):

        ## TODO: change infos & reward for tensor version?
        ## TODO: change for multiprocessing?
        rewards = []
        infos = []
        for env_indx in range(self.num_envs):
            self.steps[env_indx] += 1
            new_obs, _, done, info =  self.envs[env_indx].step(action[env_indx,:])
            rewards.append(self.reward_function(action[env_indx,:], new_obs))
            self.done[env_indx] = int(done)
            infos.append(info)

        truncation = (self.steps > self.max_episode_steps).int().reshape(self.done.shape)
        self.done = torch.maximum(self.done, truncation)
        return self.state_vector(), torch.tensor(rewards), self.done, infos


    def state_vector(self):
        for env_indx in range(self.num_envs):
            state = self.envs[env_indx]._get_full_state()
            # print("state_vector.shape = ", state.shape)
            # print("self.state[env_indx,:].shape = ", self.state[env_indx,:].shape)
            self.state[env_indx,:] = torch.from_numpy(state)[:]
        return self.state

    def render(self):
        return self.envs[0].render()

    def plot(self, ax):
        pass


@torch.no_grad()
def goal_distance(goal_a, goal_b):
    # assert goal_a.shape == goal_b.shape
    if torch.is_tensor(goal_a):
        return torch.linalg.norm(goal_a[:,:] - goal_b[:, :], axis=-1)
    else:
        return np.linalg.norm(goal_a[:, :] - goal_b[:, :], axis=-1)


@torch.no_grad()
def default_compute_reward(
        achieved_goal: Union[np.ndarray, torch.Tensor],
        desired_goal: Union[np.ndarray, torch.Tensor],
        info: dict
):
    distance_threshold = 0.1
    reward_type = "sparse"
    d = goal_distance(achieved_goal, desired_goal)
    if reward_type == "sparse":
        # if torch.is_tensor(achieved_goal):
        #     return (d < distance_threshold).double()
        # else:
        return 1.0 * (d < distance_threshold)
    else:
        return -d

class GFetchGoal(GFetch, GoalEnv, utils.EzPickle, ABC):
    def __init__(self, num_envs: int = 1):
        super().__init__(num_envs=num_envs)

        self._goal_dim = 6
        high_goal = np.ones(self._goal_dim)
        low_goal = -high_goal

        self.single_observation_space = spaces.Dict(
            dict(
                observation=self.env.action_space,
                achieved_goal=spaces.Box(
                    low_goal, high_goal, dtype=np.float64
                ),
                desired_goal=spaces.Box(
                    low_goal, high_goal, dtype=np.float64
                ),
            )
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space,
            self.num_envs)
        self.goal = None

        self.compute_reward = None
        self.set_reward_function(default_compute_reward)

        self._is_success = None
        # self.set_success_function(default_success_function)

    @torch.no_grad()
    def goal_distance(self, goal_a, goal_b):
        # assert goal_a.shape == goal_b.shape
        if torch.is_tensor(goal_a):
            return torch.linalg.norm(goal_a[:,:] - goal_b[:, :], axis=-1)
        else:
            return np.linalg.norm(goal_a[:, :] - goal_b[:, :], axis=-1)


    @torch.no_grad()
    def step(self, action):

        ## TODO: change infos & reward for tensor version?
        ## TODO: change for multiprocessing?
        infos = []
        new_obs = torch.clone(self.state)

        ## update observation (sequential)
        for env_indx in range(self.num_envs):
            _new_obs, _, done, info =  self.envs[env_indx].step(action[env_indx,:])
            new_obs[env_indx,:] = torch.from_numpy(_new_obs[:])
            infos.append(info)

        self.state = new_obs
        reward = self.reward_function(self.project_to_goal_space(self.state), self.goal, {}).reshape(
            (self.num_envs, 1))
        self.steps += 1

        truncation = (self.steps >= self.max_episode_steps).double().reshape(
            (self.num_envs, 1))

        is_success = torch.clone(reward)
        self.is_success = torch.clone(is_success)

        truncation = truncation * (1 - is_success)
        info = {'is_success': torch.clone(is_success).detach().cpu().numpy(),
                'truncation': torch.clone(truncation).detach().cpu().numpy()}
        self.done = torch.maximum(truncation, is_success)

        return (
            {
                'observation': self.state.detach().cpu().numpy(),
                'achieved_goal': self.project_to_goal_space(self.state).detach().cpu().numpy(),
                'desired_goal': self.goal.detach().cpu().numpy(),
            },
            reward.detach().cpu().numpy(),
            self.done.detach().cpu().numpy(),
            infos,
        )

    @torch.no_grad()
    def _sample_goal(self):
        # return (torch.rand(self.num_envs, 2) * 2. - 1).to(self.device)
        return ((torch.rand(self.num_envs, self._goal_dim)-0.5) * 2.0 ).to(self.device)

    @torch.no_grad()
    def reset(self, options=None, seed: Optional[int] = None, infos=None):
        self.reset_model()  # reset state to initial value
        self.goal = self._sample_goal()  # sample goal
        self.steps = torch.zeros((self.num_envs,1), dtype=torch.int).to(self.device)
        self.state = self.state_vector()
        return {
            'observation': self.state.detach().cpu().numpy(),
            'achieved_goal': self.project_to_goal_space(self.state).detach().cpu().numpy(),
            'desired_goal': self.goal.detach().cpu().numpy(),
        }

    def reset_done(self, options=None, seed: Optional[int] = None, infos=None):

        to_reset = list(torch.clone(self.done).flatten().nonzero().numpy())
        self.reset_model(env_indices = to_reset)

        newgoal = self._sample_goal()  # sample goal
        self.goal = torch.where(self.done == 1, newgoal, self.goal)

        zeros = torch.zeros((self.num_envs,1), dtype=torch.int).to(self.device)
        self.steps = torch.where(self.done==1, zeros, self.steps)
        self.state = self.state_vector()

        return {
            'observation': self.state.detach().cpu().numpy(),
            'achieved_goal': self.project_to_goal_space(self.state).detach().cpu().numpy(),
            'desired_goal': self.goal.detach().cpu().numpy(),
        }

    @torch.no_grad()
    def project_to_goal_space(self, state):
        gripper_pos = self.get_gripper_pos(state)
        object_pos = self.get_object_pos(state)

        return torch.cat((gripper_pos, object_pos), dim=-1)

    def get_gripper_pos(self, state):
        """
        get gripper position from full state
        """
        assert state.shape == (self.num_envs, 268)
        gripper_pos = state[:,84:87]

        return gripper_pos

    def get_object_pos(self, state):
        """
        get object position from full state
        """
        assert state.shape == (self.num_envs, 268)
        object_pos = state[:, 105:108]

        return object_pos

class GFetchDCIL(GFetchGoal):
    def __init__(self, device: str = 'cpu', num_envs: int = 1):
        super().__init__(num_envs)

        self.done = torch.ones((self.num_envs, 1)).int().to(self.device)

        ## fake init as each variable is modified after first reset
        self.steps = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
        self.is_success = torch.zeros((self.num_envs, 1)).int().to(self.device)
        self.goal = self.project_to_goal_space(torch.clone(self.state)).to(self.device)

        self.truncation = None

        self.max_episode_steps = torch.ones(self.num_envs, dtype=torch.int).to(self.device)*20

        self.do_overshoot = True

        self.skill_manager = SkillsManager("/Users/chenu/Desktop/PhD/github/dcil/demos/fetchenv/demo_set/1.demo", self) ## skill length in time-steps

    @torch.no_grad()
    def step(self,action):

        infos = []
        new_obs = torch.clone(self.state)

        ## update observation (sequential)
        for env_indx in range(self.num_envs):
            _new_obs, _, done, info =  self.envs[env_indx].step(action[env_indx,:])
            new_obs[env_indx,:] = torch.from_numpy(_new_obs[:])
            infos.append(info)

        self.state = new_obs

        reward = self.reward_function(self.project_to_goal_space(self.state), self.goal, {}).reshape(
            (self.num_envs, 1))
        self.steps += 1

        truncation = (self.steps >= self.max_episode_steps.view(self.steps.shape)).double().reshape(
            (self.num_envs, 1))

        is_success = torch.clone(reward)/1.
        self.is_success = torch.clone(is_success)

        truncation = truncation * (1 - is_success)
        info = {'is_success': torch.clone(is_success).detach().cpu().numpy(),
                'truncation': torch.clone(truncation).detach().cpu().numpy()}
        self.done = torch.maximum(truncation, is_success)

        ## get next goal and next goal availability boolean
        next_goal_state, info['next_goal_avail'] = self.skill_manager.next_goal()
        info['next_goal'] = self.project_to_goal_space(next_goal_state)

        return (
            {
                'observation': self.state.detach().cpu().numpy(),
                'achieved_goal': self.project_to_goal_space(self.state).detach().cpu().numpy(),
                'desired_goal': self.goal.detach().cpu().numpy(),
            },
            reward.detach().cpu().numpy(),
            self.done.detach().cpu().numpy(),
            infos,
        )

    def set_skill(self, skill_indx):
        start_state, length_skill, goal_state = self.skill_manager.set_skill(skill_indx)
        start_obs, start_sim_state = start_state

        ## set sim state for each environment
        for env_indx in range(self.num_envs):
            self.envs[env_indx].sim.set_state(start_sim_state[env_indx])

        goal = self.project_to_goal_space(goal_state)
        self.state = start_state
        zeros = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
        self.steps = zeros
        self.goal = goal

        return {
            'observation': self.state.detach().cpu().numpy(),
            'achieved_goal': self.project_to_goal_space(self.state).detach().cpu().numpy(),
            'desired_goal': self.goal.detach().cpu().numpy(),
        }

    def shift_goal(self):
        goal_state, _ = self.skill_manager.shift_goal()
        goal = self.project_to_goal_space(goal_state)
        zeros = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
        self.steps = zeros
        self.goal = goal

        return {
            'observation': self.state.detach().cpu().numpy(),
            'achieved_goal': self.project_to_goal_space(self.state).detach().cpu().numpy(),
            'desired_goal': self.goal.detach().cpu().numpy(),
        }

    @torch.no_grad()
    def _select_skill(self):
        ## done indicates indx to change
        ## overshoot indicates indx to shift by one
        ## is success indicates if we should overshoot
        return self.skill_manager._select_skill(torch.clone(self.done.int()), torch.clone(self.is_success.int()), do_overshoot = self.do_overshoot)

    @torch.no_grad()
    def reset_done(self, options=None, seed: Optional[int] = None, infos=None):

        start_state, length_skill, goal_state, b_overshoot_possible = self._select_skill()
        start_obs, start_sim_state = start_state

        ## set sim state for each environment
        for env_indx in range(self.num_envs):
            self.envs[env_indx].sim.set_state(start_sim_state[env_indx])

        goal = self.project_to_goal_space(goal_state)

        b_change_state = torch.logical_and(self.done, torch.logical_not(b_overshoot_possible)).int()
        self.state = torch.where(b_change_state == 1, start_obs, self.state)
        # self.state = torch.where(self.done == 1, start_state, self.state)
        zeros = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
        self.steps = torch.where(self.done.flatten() == 1, zeros, self.steps)
        # goal = torch.tensor(
        #     np.tile(np.array([1.78794995, 1.23542976]), (self.num_envs, 1))
        # ).to(self.device)
        self.goal = torch.where(self.done == 1, goal, self.goal).to(self.device)


        return {
            'observation': self.state.detach().cpu().numpy(),
            'achieved_goal': self.project_to_goal_space(self.state).detach().cpu().numpy(),
            'desired_goal': self.goal.detach().cpu().numpy(),
        }

    @torch.no_grad()
    def reset(self, options=None, seed: Optional[int] = None, infos=None):

        init_indx = torch.ones((self.num_envs,1)).int().to(self.device)
        start_state, length_skill, goal_state = self.skill_manager.get_skill(init_indx)
        start_obs, start_sim_state = start_state

        self.state = torch.clone(start_obs)

        goal = self.project_to_goal_space(goal_state)

        ## set sim state for each environment
        for env_indx in range(self.num_envs):
            self.envs[env_indx].sim.set_state(self.init_sim_state)

        zeros = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
        self.max_episode_steps = torch.ones(self.num_envs, dtype=torch.int).to(self.device)*10
        self.steps = zeros

        self.goal = goal

        return {
            'observation': self.state.detach().cpu().numpy(),
            'achieved_goal': self.project_to_goal_space(self.state).detach().cpu().numpy(),
            'desired_goal': self.goal.detach().cpu().numpy(),
        }


if (__name__=='__main__'):

    # demo_filename = "/Users/chenu/Desktop/PhD/github/dcil/demos/fetchenv/demo_set/1.demo"
    # L_inner_states = []
    # print("filename :\n", demo_filename)
    #
    # if not os.path.isfile(demo_filename):
    #     print ("File does not exist.")
    #
    # with open(demo_filename, "rb") as f:
    #     demo = pickle.load(f)
    #
    # for i in range(len(demo["checkpoints"])):
    #     L_inner_states.append(demo["checkpoints"][i])
    #
    # print(L_inner_states)
    #
    # for i in range(len(L_inner_states)):
    #     L_inner_states[i] = (L_inner_states[i][0],
    #                             L_inner_states[i][1],
    #                             L_inner_states[i][2],
    #                             L_inner_states[i][3][:268],
    #                             L_inner_states[i][4],
    #                             L_inner_states[i][5],
    #                             L_inner_states[i][6],
    #                             L_inner_states[i][7],
    #                             L_inner_states[i][8])

    # env = GFetch(num_envs = 2)
    #
    # obs = env.reset()
    #
    # for i in range(30):
    #     env.env.render()
    #     action = env.action_space.sample()
    #     obs, reward, done, info = env.step(action)
    #
    #     if max(done) == 1:
    #         env.reset_done()

    env = GFetchDCIL(device="cpu", num_envs = 2)

    obs = env.reset()
    print("obs = ", obs)

    print(env.compute_reward)
    print(env.project_to_goal_space)

    for i in range(30):
        env.env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        if max(done) == 1:
            env.reset_done()

        print("done = ", done)




    # new_env = MyComplexFetchEnv()
    #
    # new_env.reset()
    #
    # new_env.sim.set_state(sim_state)
    #
    # for i in range(10):
    #     new_env.env.render()
    #     action = new_env.env.action_space.sample()
    #     obs, reward, done, info = new_env.step(action)
