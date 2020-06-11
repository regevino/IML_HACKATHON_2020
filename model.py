"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2020

Authors:

===================================================
"""
import pickle
import time

from sklearn.ensemble import RandomForestClassifier

import preproccesing

CLASSIFIER_PICKLE_FILE = 'GitHubClassifierPickle'


class GitHubClassifier:

	def __init__(self, load_from_file=False):
		if not load_from_file:
			data = preproccesing.get_all_data()
			self.histogram = preproccesing.create_histogram(data)
			train_X = preproccesing.feature_creation(data.values, self.histogram)
			train_y = data['Project'].values
			self.model = RandomForestClassifier(max_depth=25)
			self.model.fit(train_X, train_y)
		else:
			with open(CLASSIFIER_PICKLE_FILE, 'rb') as pick:
				stored_classifier = pickle.load(pick)
			self.histogram = stored_classifier.histogram
			self.model = stored_classifier.model

	def classify(self, X):
		"""
		Receives a list of m unclassified pieces of code, and predicts for each
		one the Github project it belongs to.
		:param X: a numpy array of shape (m,) containing the code segments (strings)
		:return: y_hat - a numpy array of shape (m,) where each entry is a number between 0 and 6
		0 - building_tool
		1 - espnet
		2 - horovod
		3 - jina
		4 - PuddleHub
		5 - PySolFC
		6 - pytorch_geometric
		"""
		return self.model.predict(preproccesing.feature_creation(X, self.histogram))

	def store(self):
		with open(CLASSIFIER_PICKLE_FILE, 'wb') as pick:
			pickle.dump(self, pick)


if __name__ == '__main__':
	start_time = time.time()
	model = GitHubClassifier(False)
	print(model.classify(preproccesing.get_all_data()['line'].values))
	elapsed_time = time.time() - start_time
	print('Elapsed: ', elapsed_time)
	model.store()
