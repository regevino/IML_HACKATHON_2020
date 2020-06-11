import re

from pandas import DataFrame
from tqdm import tqdm
import numpy as np
from preproccesing import get_train_validate_evaluate


class BaseLine():

	def __init__(self):
		self.histogram = dict()

	def train(self, df):
		for row in tqdm(df.values, desc="Training Baseline"):
			line_of_code = row[0]
			class_of_line = row[1]
			weight_of_line = row[2]
			for word in line_of_code.split():
				word = word.strip()
				if not word:
					break
				if word not in self.histogram:
					self.histogram[word] = [0 for i in range(7)]
				self.histogram[word][class_of_line] += weight_of_line

		for word in tqdm(self.histogram, desc="Normalizing"):
			self.histogram[word] = (np.array(self.histogram[word]) - np.min(self.histogram[word])) / np.max(self.histogram[word])

	def predict(self, X):
		y_hat = []
		for sample in tqdm(X, desc="Predicting"):
			classes = [0 for i in range(7)]
			for word in sample.split():
				word = word.strip()
				if not word:
					break
				if word in self.histogram:
					word_class = np.argmax(self.histogram[word])
					classes[word_class] += self.score(word, word_class)
			y_hat.append(np.argmax(classes))
		return np.array(y_hat)

	def score(self, word, classification):
		return self.histogram[word][classification] - np.mean(self.histogram[word])


def plot_empirical_error(df: DataFrame):
	empirical_loss = np.count_nonzero(df['Project'] - df['Prediction']) / len(df['Project'])
	print(empirical_loss)


if __name__ == '__main__':
	model = BaseLine()
	train, val, eval = get_train_validate_evaluate()
	train['wieghts'] = np.ones(len(train['line']))
	model.train(train)
	print(len(model.histogram))
	y_hat = model.predict(train['line'])
	train['Prediction'] = y_hat
	plot_empirical_error(train)
	y_hat = model.predict(val['line'])
	val['Prediction'] = y_hat
	plot_empirical_error(val)
