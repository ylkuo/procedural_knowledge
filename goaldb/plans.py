# -*- coding: utf-8 -*

import re
from collections import defaultdict
from pymongo import Connection, DESCENDING
from types import *

class RawPlan():
	def __init__(self, host='abbith.media.mit.edu', port=27017):
		connection = Connection(host, port)
		db = connection['goaldb']
		self.plan_collection = db['source_plans']
		self.goal_actions_map = defaultdict(list)

	def get_actions(self, goal):
		actions = []
		goal = self.plan_collection.find_one({"goal": goal})
		actions.extend(self.get_actions_from_steps(goal['ehow']['steps']))
		actions.extend(self.get_actions_from_steps(goal['wikihow']['steps']))
		return actions

	def get_actions_from_steps(self, steps, depth=0):
		actions = []
		for step in steps:
			if type(step) is UnicodeType:
				actions.append(self.remove_wikihow_tag(step).split('.')[0])
			else:
				if depth == 0:
					actions.extend(self.get_actions_from_steps(step, depth+1))
				#else:
				#	actions.append(self.get_actions_from_steps(step, depth+1))
		return actions
	
	def populate_goal_actions_map(self, lower_bound=0, max_goals=500):
		count = 0
		for goal in self.plan_collection.find(\
				{'has_wikihow':1.0}).sort(\
				'wikihow.visitors', DESCENDING).limit(\
				max_goals)[lower_bound:max_goals]:
			if not goal.has_key('goal'):
				continue
			actions = []
			if goal.has_key('ehow'):
				actions.extend(
					self.get_actions_from_steps(goal['ehow']['steps']))
			if goal.has_key('wikihow'):
				actions.extend(
					self.get_actions_from_steps(goal['wikihow']['steps']))
			for action in actions:
				self.update_flat_goal_actions_map(goal['goal'], action)

	def update_nested_goal_actions_map(self, goal, action):
		if type(goal) is ListType:	# in order to skip some parsing errors
			return
		if type(action) is UnicodeType:
			self.goal_actions_map[goal].append(action)
		else:
			self.goal_actions_map[goal].append(action[0])
			self.update_nested_goal_actions_map(action[0], action[1])

	def update_flat_goal_actions_map(self, goal, action):
		if type(goal) is ListType:	# in order to skip some parsing errors
			return
		if type(action) is UnicodeType:
			self.goal_actions_map[goal].append(action)
		else:
			self.goal_actions_map[goal].append(action[0])
			self.update_flat_goal_actions_map(goal, action[1])

	def remove_wikihow_tag(self, data):
		p = re.compile(r'LINK\[\[\[.*?\]\]\]')
		data = p.sub('', data)
		p = re.compile(r'IMAGE\[\[\[.*?\]\]\]')
		return p.sub('', data).replace("[[[", "").replace("]]]", "").lstrip()

if __name__ == '__main__':
	plan = RawPlan()
	for action in plan.get_actions('eat a banana'):
		print action
	from util import save_pickle
	plan.populate_goal_actions_map(0, 3000)
	save_pickle('../data/goal_actions.pickle.gz', plan.goal_actions_map)
