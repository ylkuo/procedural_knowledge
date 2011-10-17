# -*- coding: utf-8 -*

import nltk
import numpy as np
from collections import defaultdict
from nltk.corpus import wordnet as wn
from plans import RawPlan
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from util import load_pickle, cosine_similarity, flatten, edit_distance

class GoalCluster():
	def __init__(self, goal_path=None):
		self.plan = RawPlan()
		self.goal_list = []
		self.goal_vector = []
		self.stemmer = nltk.PorterStemmer()
		if goal_path is None:
			self.plan.populate_goal_actions_map()
			self.goal_actions_map = self.plan.goal_actions_map
			self.goal_list = self.goal_actions_map.keys()
		else:
			self.goal_actions_map = load_pickle(goal_path)
			self.goal_list = self.goal_actions_map.keys()

	def create_tfidf_vector(self):
		count_vect = CountVectorizer()
		doc = map(lambda x: " ".join(flatten(x)) + " " + \
				x[0], self.goal_actions_map.items())
		X_train_counts = count_vect.fit_transform(doc)
		tfidf_transformer = TfidfTransformer()
		X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
		return X_train_tfidf

	def clustering(self):
		# Calculate similarity matrix
		X = self.create_tfidf_vector()
		X = X.toarray()
		pca = PCA(n_components=300, copy=False)
		X = pca.fit(X).transform(X)
		S = cosine_similarity(X, X)
		# Run affinity propogation
		af = AffinityPropagation()
		af.fit(S)
		# Formulate result
		tmp_clusters = defaultdict(list)
		goal_clusters = defaultdict(list)
		cluster_centers_indices = af.cluster_centers_indices_
		labels = af.labels_
		count = 0
		for label in labels:
			tmp_clusters[\
				self.goal_list[cluster_centers_indices[label]]].append(\
				self.goal_list[count])
			count += 1
		# 2nd-layer clutering of each cluster
		for goal, item_list in tmp_clusters.items():
			subclusters = self.subcluster_by_editdistance(goal, item_list)
			for subgoal, items in subclusters.items():
				goal_clusters[subgoal] = items
		return goal_clusters

	def subcluster_by_editdistance(self, center, item_list, threshold=2):
		clusters = defaultdict(list)
		clusters[center].append(center)
		for item in item_list:
			if item in clusters.keys():
				continue
			flag = 0
			list_item = self.stemmer.stem(item.encode('utf-8')).split()
			for goal in clusters.keys():
				list_goal = self.stemmer.stem(goal.encode('utf-8')).split()
				d = edit_distance(list_goal, list_item)
				if d < threshold:
					clusters[goal].append(item)
					flag = 1
					break
			if flag == 0:
				clusters[item].append(item)
		return clusters

	def get_wordnet_lch(self, concept1, concept2, max_depth=5):
		if self.stemmer.stem(concept1) == self.stemmer.stem(concept2):
			return concept1
		concept1_synsets = wn.synsets(concept1)
		concept2_synsets = wn.synsets(concept2)
		for concept1_synset in concept1_synsets:
			for concept2_synset in concept2_synsets:
				commons = \
					concept1_synset.lowest_common_hypernyms(concept2_synset)
				for common in commons:
					print common
					if common.max_depth() > max_depth:
						return common.lemma_names[0].replace('_', ' ')
		return None

	def get_generalized_goal(self, center, goal_list):
		output = []
		center_tokens = nltk.word_tokenize(center.encode('utf-8'))
		print center_tokens
		for goal in goal_list:
			if goal == center:
				continue
			goal_tokens = nltk.word_tokenize(goal.encode('utf-8'))
			print goal_tokens
			goal_count = 0
			center_count = 0
			while center_count < len(center_tokens):
				match = False
				token = center_tokens[center_count]
				print token
				while match == False:
					if center_count >= len(center_tokens) or \
							goal_count >= len(goal_tokens):
						break
					print goal_tokens[goal_count]
					print goal_count
					print center_count
					common = self.get_wordnet_lch(\
								token, goal_tokens[goal_count])
					if common != None:
						output.append(common)
						match = True
						goal_count += 1
						center_count += 1
					elif center_count < len(center_tokens) - 1:
						if goal_count < len(goal_tokens) - 1:
							if self.get_wordnet_lch(\
									center_tokens[center_count+1], \
									goal_tokens[goal_count+1]) != None:
								match = True
								goal_count += 1
								center_count += 1
							elif self.get_wordnet_lch(\
									center_tokens[center_count], \
									goal_tokens[goal_count+1]) != None:
								match = True
								goal_count += 1
							elif self.get_wordnet_lch(\
									center_tokens[center_count+1], \
									goal_tokens[goal_count]) != None:
								match = True
								center_count += 1
							else:
								break
						else:
							if self.get_wordnet_lch(\
									center_tokens[center_count+1], \
									goal_tokens[goal_count]) != None:
								match = True
								center_count += 1
							else:
								break
					elif goal_count < len(goal_tokens) - 1:
						if self.get_wordnet_lch(\
								center_tokens[center_count], \
								goal_tokens[goal_count+1]) != None:
							match = True
							goal_count += 1
						else:
							break
					else:
						break
				if match == False and \
						(goal_count == len(goal_tokens)-1 or \
						center_count == len(center_tokens)-1):
					break
				elif match == False:
					output = []
					goal_count = 0
					center_count = 0
					break	
			if len(output) > 0:
				if output[-1] in ['a', 'an', 'the', 'be']:
					del output[-1]
				return " ".join(output)
		return None

if __name__ == '__main__':
	cluster = GoalCluster('../data/goal_actions.pickle.gz')
	goal_clusters = cluster.clustering()
	fp = open('../data/output_cluster.txt', 'w')
	for center, cluster_items in goal_clusters.items():
		if len(cluster_items) > 1:
			output = cluster.get_generalized_goal(center, cluster_items)
			if output != None:
				cluster_list = ",".join(cluster_items)
				fp.write(output + '|' + cluster_list.encode('utf-8') + '\n')
