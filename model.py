"""
===================================================
	 Introduction to Machine Learning (67577)
			 IML HACKATHON, June 2020

Authors:

===================================================
"""
import gzip
import pickle
from sklearn.ensemble import RandomForestClassifier

import preproccesing

CLASSIFIER_PICKLE_FILE = 'GitHubClassifierPickle'


class GitHubClassifier:

	def __init__(self, load_from_file=True):
		"""
		Craete a classifier. by default, load the classifier from a file
		:param load_from_file: Create a new classifier if False, otherwise unpickle form file (default: True)
		"""
		if not load_from_file:
			data = preproccesing.get_all_data()
			self.histogram = preproccesing.create_histogram(data)
			self.train_X = preproccesing.feature_creation(data['line'].values, self.histogram)
			self.train_y = data['Project'].values
		else:
			with gzip.open(CLASSIFIER_PICKLE_FILE, 'rb') as pick:
				stored_classifier = pickle.load(pick)
			self.histogram = stored_classifier.histogram
			self.train_X = stored_classifier.train_X
			self.train_y = stored_classifier.train_y

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
		model = RandomForestClassifier(max_depth=25)
		model.fit(self.train_X, self.train_y)
		return model.predict(preproccesing.feature_creation(X, self.histogram))

	def store(self):
		"""
		Store this classifier in a file. Forest is not stored because it is very large and have enough time
		to fit it before prediction.
		:return:
		"""
		with gzip.open(CLASSIFIER_PICKLE_FILE, 'wb') as pick:
			pickle.dump(self, pick)
