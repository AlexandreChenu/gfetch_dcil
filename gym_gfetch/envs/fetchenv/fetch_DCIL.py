
# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.

# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from .import_ai import *
# from import_ai import *

from abc import ABC
from abc import abstractmethod
from typing import Optional

from .fetch_env import ComplexFetchEnv
# from fetch_env import ComplexFetchEnv

from typing import Union
from gym import utils, spaces
from gym import error
import numpy as np
import torch
from matplotlib import collections as mc
# from IPython import embed

from .skill_manager_fetchenv import SkillsManager
# from skill_manager_fetchenv import SkillsManager

import gym
gym._gym_disable_underscore_compat = True

import types
os.environ["PATH"] = os.environ["PATH"].replace('/usr/local/nvidia/bin', '')
# try:
import mujoco_py

from gym.envs.mujoco import mujoco_env
# except Exception:
    # print('WARNING: could not import mujoco_py. This means robotics environments will not work')
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
class GFetch(mujoco_env.MujocoEnv, utils.EzPickle, ABC):
    TARGET_SHAPE = 0
    MAX_PIX_VALUE = 0

    def __init__(self, model_file='teleOp_boxes_1.xml', nsubsteps=20, min_grip_score=0, max_grip_score=0,
                 target_single_shelf=False, combine_table_shelf_box=False, ordered_grip=False,
                 target_location='1000', timestep=0.002, force_closed_doors=False):


        self.env = ComplexFetchEnv(
            model_file=model_file, nsubsteps=nsubsteps,
            min_grip_score=min_grip_score, max_grip_score=max_grip_score,
            #ret_full_state=False,
            ret_full_state=True,
            target_single_shelf=target_single_shelf,
            combine_table_shelf_box=combine_table_shelf_box, ordered_grip=ordered_grip,
            target_location=target_location, timestep=timestep,
            force_closed_doors=force_closed_doors
        )

        self.action_space = self.env.action_space

        self.observation_space = self.env.observation_space

        self.set_reward_function(default_reward_fun)

        init_state = self.env.reset()
        self.init_state = init_state.copy()

        self.init_sim_state = self.env.sim.get_state()
        self.init_qpos = self.init_sim_state.qpos.copy()
        self.init_qvel = self.init_sim_state.qvel.copy()


        self.state = self.init_state.copy()
        self.done = False
        self.steps = 0

        self.max_episode_steps = 10

        self.rooms = []

    def __getattr__(self, e):
        assert self.env is not self
        return getattr(self.env, e)

    def set_reward_function(self, reward_function):
        self.compute_reward = (
            reward_function  # the reward function is not defined by the environment
        )

    def reset_model(self, env_indices=None) -> np.ndarray:
        """
        Reset environments to initial simulation state & return vector state
        """
        self.env.sim.set_state(self.init_sim_state)

        return self.state_vector()

    def reset(self, options=None, seed: Optional[int] = None, infos=None):
        self.reset_model()
        self.steps = 0
        return self.state_vector()

    def reset_done(self, options=None, seed: Optional[int] = None, infos=None):
        self.reset_model()
        self.steps = 0
        return self.state_vector()

    def step(self, action):
        self.steps += 1
        new_obs, _, done, info =  self.env.step(action)
        reward = self.compute_reward(action, new_obs)
        self.done = done
        truncation = (self.steps > self.max_episode_steps)
        self.done = (self.done or truncation)
        return self.state_vector(), reward, self.done, info


    def state_vector(self):
        state = self.env._get_full_state()
        self.state = state.copy()
        return self.state

    def render(self):
        return self.env.render()

    def plot(self, ax):
        pass


@torch.no_grad()
def goal_distance(goal_a, goal_b):
    # assert goal_a.shape == goal_b.shape
    #print("\ngoal_a = ", goal_a)
    #print("goal_b = ", goal_b)
    #print("d = ", np.linalg.norm(goal_a - goal_b, axis=-1))
    if torch.is_tensor(goal_a):
        return torch.linalg.norm(goal_a - goal_b, axis=-1)
    else:
        return np.linalg.norm(goal_a - goal_b, axis=-1)


@torch.no_grad()
def default_compute_reward(
        achieved_goal: Union[np.ndarray, torch.Tensor],
        desired_goal: Union[np.ndarray, torch.Tensor],
        info: dict
):
    distance_threshold = 0.05
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
    def __init__(self):
        super().__init__()

        self._goal_dim = 6 ## TODO: set automatically
        high_goal = np.ones(self._goal_dim)
        low_goal = -high_goal

        self.observation_space = spaces.Dict(
            dict(
                observation=self.env.observation_space,
                achieved_goal=spaces.Box(
                    low_goal, high_goal, dtype=np.float64
                ),
                desired_goal=spaces.Box(
                    low_goal, high_goal, dtype=np.float64
                ),
            )
        )

        self.goal = None

        self.compute_reward = None
        self.set_reward_function(default_compute_reward)

        self._is_success = None
        # self.set_success_function(default_success_function)

    @torch.no_grad()
    def goal_distance(self, goal_a, goal_b):
        # assert goal_a.shape == goal_b.shape
        if torch.is_tensor(goal_a):
            return torch.linalg.norm(goal_a - goal_b, axis=-1)
        else:
            return np.linalg.norm(goal_a - goal_b, axis=-1)


    @torch.no_grad()
    def step(self, action):
        self.steps += 1
        cur_state = self.state.copy()

        new_state, _, done, info =  self.env.step(action)
        reward = self.compute_reward(self.project_to_goal_space(new_state), self.goal, {})

        truncation = (self.steps >= self.max_episode_steps)

        is_success = reward.copy()
        self.is_success = is_success.copy()

        truncation = truncation * (1 - is_success)
        info = {'is_success': is_success,
                'truncation': truncation}
        self.done = (done or bool(truncation)) or bool(is_success)

        return (
            {
                'observation': self.state,
                'achieved_goal': self.project_to_goal_space(self.state),
                'desired_goal': self.goal,
            },
            reward,
            self.done,
            info,
        )

    @torch.no_grad()
    def _sample_goal(self):
        # return (torch.rand(self.num_envs, 2) * 2. - 1).to(self.device)
        return np.random.uniform(-1.,1., size=self._goal_dim)

    @torch.no_grad()
    def reset(self, options=None, seed: Optional[int] = None, infos=None):
        self.reset_model()  # reset state to initial value
        self.goal = self._sample_goal()  # sample goal
        self.steps = 0
        self.state = self.state_vector()
        return {
            'observation': self.state,
            'achieved_goal': self.project_to_goal_space(self.state),
            'desired_goal': self.goal,
        }

    def reset_done(self, options=None, seed: Optional[int] = None, infos=None):

        self.reset_model()

        newgoal = self._sample_goal()  # sample goal
        self.goal = newgoal.copy()

        self.steps = 0.
        self.state = self.state_vector()

        return {
            'observation': self.state,
            'achieved_goal': self.project_to_goal_space(self.state),
            'desired_goal': self.goal,
        }

    @torch.no_grad()
    def project_to_goal_space(self, state):
        gripper_pos = self.get_gripper_pos(state)
        object_pos = self.get_object_pos(state)

        return np.concatenate((gripper_pos, object_pos), axis=-1)
        # return gripper_pos

    def get_gripper_pos(self, state):
        """
        get gripper position from full state
        """
        assert state.shape == (268,)
        gripper_pos = state[84:87]

        return gripper_pos

    def get_object_pos(self, state):
        """
        get object position from full state
        """
        assert state.shape == (268,)
        object_pos = state[105:108]

        return object_pos


class GFetchDCIL(GFetchGoal):
    def __init__(self, demo_path):
        super().__init__()

        self.done = False
        ## fake init as each variable is modified after first reset
        self.steps = 0
        self.is_success = 0
        self.goal = None

        self.truncation = None
        self.max_episode_steps = 50
        self.do_overshoot = True

        self.demo_path = demo_path
        self.skill_manager = SkillsManager(self.demo_path, self) ## skill length in time-steps


    @torch.no_grad()
    def step(self,action):

        infos = []
        cur_state = self.state.copy()

        ## update observation (sequential)
        new_state, _, done, info =  self.env.step(action)
        infos.append(info)

        self.state = new_state.copy()

        reward = self.compute_reward(self.project_to_goal_space(self.state), self.goal, {})
        self.steps += 1

        truncation = (self.steps >= self.max_episode_steps)

        is_success = reward.copy()
        self.is_success = is_success.copy()

        truncation = truncation * (1 - is_success)
        info = {'is_success': is_success,
                'truncation': truncation}

        # print("\nis_success = ", is_success)
        # print("truncation = ", truncation)
        # print("done from step = ", done)
        # self.done = (done or bool(truncation)) or bool(is_success)
        self.done = bool(truncation) or bool(is_success)
        # print("self.done = ", self.done)

        ## get next goal and next goal availability boolean
        next_goal_state, info['next_goal_avail'] = self.skill_manager.next_goal()
        info['next_goal'] = self.project_to_goal_space(next_goal_state)

        return (
            {
                'observation': self.state.copy(),
                'achieved_goal': self.project_to_goal_space(self.state).copy(),
                'desired_goal': self.goal.copy(),
            },
            reward,
            self.done,
            info,
        )

    def set_skill(self, skill_indx):
        start_state, length_skill, goal_state = self.skill_manager.set_skill(skill_indx)
        start_obs, start_sim_state = start_state

        ## set sim state for each environment
        self.env.sim.set_state(start_sim_state)

        goal = self.project_to_goal_space(goal_state)
        self.state = start_state.copy()
        self.steps = 0
        self.goal = goal

        return {
            'observation': self.state.copy(),
            'achieved_goal': self.project_to_goal_space(self.state).copy(),
            'desired_goal': self.goal.copy(),
        }

    def shift_goal(self):
        goal_state, _ = self.skill_manager.shift_goal()
        goal = self.project_to_goal_space(goal_state)
        self.steps = 0
        self.goal = goal

        return {
            'observation': self.state.copy(),
            'achieved_goal': self.project_to_goal_space(self.state).copy(),
            'desired_goal': self.goal.copy(),
        }

    @torch.no_grad()
    def _select_skill(self):
        ## done indicates indx to change
        ## overshoot indicates indx to shift by one
        ## is success indicates if we should overshoot
        return self.skill_manager._select_skill(self.done, self.is_success, do_overshoot = self.do_overshoot)

    @torch.no_grad()
    def reset_done(self, options=None, seed: Optional[int] = None, infos=None):

        _start_state, length_skill, goal_state, overshoot_possible = self._select_skill()
        start_state, start_sim_state = _start_state

        if self.done:
            if self.is_success:
                self.skill_manager.add_success(self.skill_manager.indx_goal)
            else:
                self.skill_manager.add_failure(self.skill_manager.indx_goal)

        ## reset robot to known state if no overshoot possible
        if not (self.is_success and self.do_overshoot and overshoot_possible):
            self.env.sim.set_state(start_sim_state)
            self.state = start_state.copy()

        goal = self.project_to_goal_space(goal_state)
        self.goal = goal.copy()

        self.steps = 0
        self.max_episode_steps = length_skill

        #achieved_goal = self.project_to_goal_space(self.state)
        # print("achieved goal from reset_done = ", achieved_goal)
        # sys.stdout.flush()

        return {
            'observation': self.state.copy(),
            'achieved_goal': self.project_to_goal_space(self.state).copy(),
            'desired_goal': self.goal.copy(),
        }

    @torch.no_grad()
    def reset(self, options=None, seed: Optional[int] = None, infos=None):

        init_indx = 1
        _start_state, length_skill, goal_state = self.skill_manager.get_skill(init_indx)
        start_state, start_sim_state = _start_state

        ## set sim state for each environment
        self.env.sim.set_state(start_sim_state)
        self.state = start_state.copy()

        self.max_episode_steps = length_skill
        self.steps = 0

        goal = self.project_to_goal_space(goal_state)
        self.goal = goal.copy()

        achieved_goal = self.project_to_goal_space(self.state)
        # print("achieved goal from reset = ", achieved_goal)
        # sys.stdout.flush()

        return {
            'observation': self.state.copy(),
            'achieved_goal': self.project_to_goal_space(self.state).copy(),
            'desired_goal': self.goal.copy(),
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

    env = GFetchDCIL()

    obs = env.reset()
    print("obs = ", obs)

    for i in range(30):
        env.env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        if done:
            env.reset_done()

    print("obs = ", obs)
    print("done = ", done)
    print("info = ", info)

    # env = GFetchDCIL(device="cpu", num_envs = 2)
    #
    # obs = env.reset()
    # print("obs = ", obs)
    #
    # print(env.compute_reward)
    # print(env.project_to_goal_space)
    #
    # for i in range(30):
    #     env.env.render()
    #     action = env.action_space.sample()
    #     obs, reward, done, info = env.step(action)
    #
    #     if max(done) == 1:
    #         env.reset_done()
    #
    #     print("done = ", done)




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
