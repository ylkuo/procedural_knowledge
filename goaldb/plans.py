# -*- coding: utf-8 -*

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
				actions.append(step.split('.')[0])
			else:
				if depth == 0:
					actions.extend(self.get_actions_from_steps(step, depth+1))
				else:
					actions.append(self.get_actions_from_steps(step, depth+1))
		return actions
	
	def populate_goal_actions_map(self):
		count = 0
		for goal in self.plan_collection.find(
				{'has_wikihow':1.0}).sort(
				'wikihow.visitors', DESCENDING).limit(500):
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
				self.update_goal_actions_map(goal['goal'], action)

	def update_goal_actions_map(self, goal, action):
		if type(goal) is ListType:	# in order to skip some parsing errors
			return
		if type(action) is UnicodeType:
			self.goal_actions_map[goal].append(action)
		else:
			self.goal_actions_map[goal].append(action[0])
			self.update_goal_actions_map(action[0], action[1])

if __name__ == '__main__':
	plan = RawPlan()
	for action in plan.get_actions('travel around the world'):
		print action
	from util import save_pickle
	plan.populate_goal_actions_map()
	save_pickle('../data/goal_actions.pickle.gz', plan.goal_actions_map)
