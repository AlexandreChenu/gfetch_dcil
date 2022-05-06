import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import math
import random
import copy
from collections import OrderedDict
import torch
import pickle



class SkillsManager():

	def __init__(self, demo_path, env):

		self.env = env

		self.eps_state = 0.5 ## threshold distance in goal space for skill construction
		self.beta = 2.

		self.L_full_demonstration = self.extract_from_demo(demo_path)
		for i in range(len(self.L_full_demonstration)):
			self.L_full_demonstration[i] = (self.L_full_demonstration[i][0],
									self.L_full_demonstration[i][1],
									self.L_full_demonstration[i][2],
									self.L_full_demonstration[i][3][:268],
									self.L_full_demonstration[i][4],
									self.L_full_demonstration[i][5],
									self.L_full_demonstration[i][6],
									self.L_full_demonstration[i][7],
									self.L_full_demonstration[i][8])

		self.L_states, self.L_sim_states, self.L_budgets = self.clean_demo(self.L_full_demonstration)

		self.L_states = self.L_states[-4:]
		self.L_sim_states = self.L_sim_states[-4:]
		self.L_budgets = self.L_budgets[-3:]

		# print("self.L_budgets = ", self.L_budgets)

		self.nb_skills = len(self.L_states)-1

		## init indx for start and goal states
		self.indx_start = 0
		self.indx_goal = 1

		## a list of list of results per skill
		self.L_skills_results = [[] for _ in self.L_states]
		self.L_overshoot_results = [[[]] for _ in self.L_states]

		self.skill_window = 20
		self.max_size_starting_state_set = 100

		self.weighted_sampling = False

		self.delta_step = 1
		self.dist_threshold = 0.1

	def extract_from_demo(self, demo_path, verbose=0):
		"""
		Extract demo from path
		"""
		L_inner_states = []

		assert os.path.isfile(demo_path)

		with open(demo_path, "rb") as f:
			demo = pickle.load(f)
		for inner_state in zip(demo["checkpoints"]):
			L_inner_states.append(inner_state[0])
		return L_inner_states

	def clean_demo(self, L_inner_states):
		"""
		Clean the demonstration trajectory according to hyperparameter eps_dist.
		"""
		self.env.env.set_inner_state(L_inner_states[0])
		L_states_clean = [self.env.state_vector()]
		L_sim_states_clean = [self.env.env.sim.get_state()]
		L_budgets = []

		i = 0
		while i < len(L_inner_states)-1:
			k = 1
			sum_dist = 0

			# cumulative distance
			while sum_dist <= self.eps_state and i + k < len(L_inner_states) - 1:
				self.env.env.set_inner_state(L_inner_states[i+k-1])
				prev_state = self.env.state_vector().copy()
				prev_sim_state = copy.deepcopy(self.env.env.sim.get_state())
				self.env.env.set_inner_state(L_inner_states[i+k])
				new_state = self.env.state_vector().copy()
				new_sim_state = copy.deepcopy(self.env.env.sim.get_state())

				sum_dist += self.env.goal_distance(self.env.project_to_goal_space(new_state), self.env.project_to_goal_space(prev_state))
				k += 1
			if sum_dist > self.eps_state or i + k == len(L_inner_states) - 1:
				L_states_clean.append(new_state)
				# L_states_clean.append(torch.tensor(np.tile(np.array(new_state), (self.num_envs, 1))))
				L_sim_states_clean.append(new_sim_state)

			# L_budgets.append(int(self.beta*k))
			L_budgets.append(int(self.beta*k))

			i = i + k

		return L_states_clean, L_sim_states_clean, L_budgets


	def get_skill(self, skill_indx):
		"""
		Get starting state, length and goal associated to a given skill
		"""
		assert skill_indx > 0
		assert skill_indx < len(self.L_states)

		indx_start = skill_indx - self.delta_step
		indx_goal = skill_indx

		length_skill = self.L_budgets[indx_start]

		starting_state = self.get_starting_state(indx_start)
		goal_state = self.get_goal_state(indx_goal)

		return starting_state, length_skill, goal_state

	def set_skill(self, skill_indx):
		"""
		Get starting state, length and goal associated to a given skill
		"""
		assert skill_indx > 0
		assert skill_indx < len(self.L_states)

		self.indx_start = skill_indx - self.delta_step
		self.indx_goal = skill_indx

		length_skill = self.budgets[self.indx_start]

		starting_state = self.get_starting_state(self.indx_start)
		goal_state = self.get_goal_state(self.indx_goal)

		return starting_state, length_skill, goal_state

	def get_starting_state(self, indx_start):

		return self.L_states[indx_start], self.L_sim_states[indx_start]

	def get_goal_state(self, indx_goal):
		# print("indx_goal = ", indx_goal)
		return self.L_states[indx_goal]
	#
	# def add_success_overshoot(self,skill_indx):
	# 	self.L_overshoot_feasible[skill_indx-1]=True
	# 	return

	def add_success(self, skill_indx):
		"""
		Monitor successes for a given skill (to bias skill selection)
		"""
		self.L_skills_results[skill_indx].append(1)

		if len(self.L_skills_results[skill_indx]) > self.skill_window:
			self.L_skills_results[skill_indx].pop(0)

		return

	def add_failure(self, skill_indx):
		"""
		Monitor failures for a given skill (to bias skill selection)
		"""
		self.L_skills_results[skill_indx].append(0)

		if len(self.L_skills_results[skill_indx]) > self.skill_window:
			self.L_skills_results[skill_indx].pop(0)

		return

	def get_skill_success_rate(self, skill_indx):

		nb_skills_success = self.L_skills_results[skill_indx].count(1)
		s_r = float(nb_skills_success/len(self.L_skills_results[skill_indx]))

		## keep a small probability for successful skills to be selected in order
		## to avoid catastrophic forgetting
		if s_r <= 0.1:
			s_r = 10
		else:
			s_r = 1./s_r

		return s_r

	def get_skills_success_rates(self):

		L_rates = []
		for i in range(self.delta_step, len(self.L_states)):
			L_rates.append(self.get_skill_success_rate(i))

		return L_rates


	def sample_skill_indx(self):
		"""
		Sample a skill indx.

		2 cases:
			- weighted sampling of skill according to skill success rates
			- uniform sampling
		"""
		weights_available = True
		for i in range(self.delta_step,len(self.L_skills_results)):
			if len(self.L_skills_results[i]) == 0:
				weights_available = False

		## fitness based selection
		if self.weighted_sampling and weights_available:

			L_rates = self.get_skills_success_rates()

			assert len(L_rates) == len(self.L_states) - self.delta_step

			## weighted sampling
			total_rate = sum(L_rates)


			L_new_skill_indx = []
			for i in range(self.num_envs):
				pick = random.uniform(0, total_rate)

				current = 0
				for i in range(0,len(L_rates)):
					s_r = L_rates[i]
					current += s_r
					if current > pick:
						break

				i += self.delta_step
				L_new_skill_indx.append([i])

			## TODO: switch for tensor version
			new_skill_indx = torch.tensor(L_new_skill_indx)

			assert new_skill_indx.shape == (self.num_envs, 1)

		## uniform sampling
		else:
			new_skill_indx = np.random.randint(1, self.nb_skills+1)

		return new_skill_indx

	def shift_goal(self):
		"""
		Returns next goal state corresponding
		"""
		next_skill_indx = self.indx_goal + 1
		next_skill_avail = (next_skill_indx <= self.nb_skills)
		if next_skill_avail:
			self.indx_goal = next_skill_indx
		next_goal_state = self.get_goal_state(self.indx_goal)

		return next_goal_state, next_skill_avail

	def next_goal(self):
		"""
		Returns next goal state corresponding
		"""
		next_skill_indx = self.indx_goal + 1
		next_skill_avail = (next_skill_indx <= self.nb_skills)

		if next_skill_avail:
			next_goal_state = self.get_goal_state(next_skill_indx)
		else:
			next_goal_state = self.get_goal_state(self.indx_goal)

		return next_goal_state, next_skill_avail


	def next_skill_indx(self, cur_indx):
		"""
		Shift skill indices by one and assess if skill indices are available
		"""

		next_indx = cur_indx + 1
		next_skill_avail = (next_indx <= self.nb_skills)

		return next_indx, next_skill_avail

	def _select_skill(self, done, is_success, init=False, do_overshoot = True):
		"""
		Select skills (starting state, budget and goal)
		"""

		sampled_skill_indx = self.sample_skill_indx() ## return tensor of new indices
		next_skill_indx, next_skill_avail = self.next_skill_indx(self.indx_goal) ## return tensor of next skills indices

		if is_success and do_overshoot and next_skill_avail:
			self.indx_goal = next_skill_indx
		else:
			self.indx_goal = sampled_skill_indx

		## skill indx coorespond to a goal indx
		self.indx_start = (self.indx_goal - self.delta_step)

		length_skill = self.L_budgets[self.indx_start]

		starting_state = self.get_starting_state(self.indx_start)
		goal_state = self.get_goal_state(self.indx_goal)

		return starting_state, length_skill, goal_state, next_skill_avail


	def _random_skill(self):
		"""
		Select skills (starting state, budget and goal)
		"""

		sampled_skill_indx = self.sample_skill_indx() ## return tensor of new indices

		self.indx_goal = sampled_skill_indx

		## skill indx coorespond to a goal indx
		self.indx_start = (self.indx_goal - self.delta_step)
		# print("self.indx_start = ", self.indx_start.view(-1))
		length_skill = self.L_budgets[self.indx_start]

		starting_state = self.get_starting_state(self.indx_start)
		goal_state = self.get_goal_state(self.indx_goal)

		return starting_state, length_skill, goal_state



if (__name__=='__main__'):
	from fetch_DCIL import GFetchDCIL

	# env = GMazeCommon(device="cpu", num_envs=2)
	#
	# state = env.state
	# new_state = torch.tensor([[0.6000, 0.6000, 0.0000],
	#                           [-0.1000, 0.1000, 0.0000]])
	#
	# env.valid_step(state, new_state)

	traj = []

	env = GFetchDCIL()
	env.reset()

	demo_path = "/Users/chenu/Desktop/PhD/github/dcil/demos/fetchenv/demo_set/1.demo"
	sm = SkillsManager(demo_path, env)

	print("length full demo = ", len(sm.L_full_demonstration))
	print("length cleaned demo = ", len(sm.L_states))
	# print("length budgets = ", len(sm.L_budgets))

	# # print("states = ", sm.states)
	# print("states.shape = ", sm.states.shape)
	# # # print("budget = ", sm.budgets)
	# print("budget.shape = ", sm.budgets.shape)
	# #
	# # sm._random_skill()
	# #
	print("start indx = ", sm.indx_start)
	print("goal indx = ", sm.indx_goal)



	next_skill_indx, next_skill_avail = sm.next_skill_indx(sm.indx_goal)
	#
	print("next skill indx = ", next_skill_indx)
	print("next skill avail = ", next_skill_avail)
	#
	sampled_skill_indx = sm.sample_skill_indx()
	# #
	print("sampled skill indx = ", sampled_skill_indx)
	# #
	# # print("states = ", sm.states[:,0,:])
	# #
	#
	# skill_indx = torch.ones((sm.num_envs,1)).int().to(sm.device)*14
	#
	start_state, budget, goal_state = sm.get_skill(sampled_skill_indx)
	# start_state, budget, goal_state = sm.get_skill(sampled_skill_indx)
	# start_state, budget, goal_state = sm.set_skill(skill_indx)

	print("budget = ", budget)
	print("goal_state = ", goal_state)
	print("start_state = ", start_state)

	# next_goal_state, next_skill_avail = sm.next_goal()
	# print("next_goal_state = ", next_goal_state)
	# print("next_skill_avail = ", next_skill_avail)

	# print(env.state)
