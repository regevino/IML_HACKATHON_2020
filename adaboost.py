"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Gad Zalcberg
Date: February, 2019

"""
import sys

import numpy as np
from pandas import DataFrame
from matplotlib import pyplot as plt
# from plotnine import *
import tqdm
from preproccesing import get_train_validate_evaluate
from baseline import BaseLine


class AdaBoost(object):

	def __init__(self, WL, T):
		"""
		Parameters
		----------
		WL : the class of the base weak learner
		T : the number of base learners to learn
		"""
		self.WL = WL
		self.T = T
		self.h = [None] * T  # list of base learners
		self.w = np.zeros(T)  # weights

	def train(self, df: DataFrame):
		"""
		Parameters
		----------
		X : samples, shape=(num_samples, num_features)
		y : labels, shape=(num_samples)
		Train this classifier over the sample (X,y)
		After finish the training return the weights of the samples in the last iteration.
		"""
		X = df['line']
		y = df['Project']

		# sample size
		m = len(y)

		# initilize arrays
		samples_weights = np.zeros(shape=(self.T + 1, m))

		# initialize weights (for 0's iteration) uniformly
		samples_weights[0] = np.ones(shape=m) / m
		stump_pred = []
		for t in tqdm.tqdm(range(self.T), desc="Training Model"):
			# fit  weak learner
			curr_samples_weights = samples_weights[t]
			self.h[t] = self.WL()
			df['weights'] = curr_samples_weights
			self.h[t].train(df)

			# calculate error and stump weight from weak learner prediction
			old_stump_pred = stump_pred
			stump_pred = self.h[t].predict(X)
			assert not (np.array_equal(old_stump_pred, stump_pred))

			err = curr_samples_weights[(stump_pred != y)].sum()
			stump_weight = np.log((1 - err) / err) / 2

			# update sample weights
			new_samples_weights = curr_samples_weights * np.exp(-stump_weight * y * stump_pred)
			new_samples_weights /= new_samples_weights.sum()

			# update sample weights for t+1
			samples_weights[t + 1] = new_samples_weights

			# save results of iteration
			self.w[t] = stump_weight
		return samples_weights[-1]

	def predict(self, X, max_t):
		"""
		Parameters
		----------
		X : samples, shape=(num_samples, num_features)
		:param max_t: integer < self.T: the number of classifiers to use for the classification
		:return: y_hat : a prediction vector for X. shape=(num_samples)
		Predict only with max_t weak learners,
		"""
		prediction = [self.h[i].predict(X) for i in
					  tqdm.tqdm(range(max_t), desc=f"Predicting for {max_t} iterations")]
		prediction = np.transpose(prediction)
		prediction = [np.bincount(prediction[i], minlength=7, weights=self.w) for i in
					  range(len(prediction))]
		return np.argmax(prediction, axis=1)

	def error(self, X, y, max_t):
		"""
		Parameters
		----------
		X : samples, shape=(num_samples, num_features)
		y : labels, shape=(num_samples)
		:param max_t: integer < self.T: the number of classifiers to use for the classification
		:return: error : the ratio of the correct predictions when predict only with max_t weak learners (float)
		"""
		result = self.predict(X, max_t) != y
		return np.mean(result)


if __name__ == '__main__':
	num_to_boost = int(sys.argv[1])
	train, val, eval = get_train_validate_evaluate()
	model = AdaBoost(BaseLine, num_to_boost)
	model.train(train)
	print(model.error(val['line'].values, val['Project'].values, num_to_boost))
