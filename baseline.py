from pandas import DataFrame
from tqdm import tqdm
import numpy as np
from preproccesing import get_train_validate_evaluate


# This file contains out initial baseline learner, with some improvements we made along the way.

class BaseLine():

	def __init__(self):
		"""
		Create baseline learner.
		"""
		self.histogram = dict()

	def train(self, df):
		"""
		Train learner on the data (create histogram).
		:param df: DAtaFrame containing data.
		"""
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
			classification = np.argmax(self.histogram[word])
			score = self.score(word, classification)
			self.histogram[word] = dict()
			self.histogram[word]['class'] = classification
			self.histogram[word]['score'] = score

	def predict(self, X):
		"""
		Create a prediction with base learner.
		:param X: Data to predict for.
		:return: numpy array containing prediction labels.
		"""
		y_hat = []
		for sample in tqdm(X, desc="Predicting"):
			classes = [0 for i in range(7)]
			for word in sample.split():
				word = word.strip()
				if not word:
					break
				if word in self.histogram:
					classes[self.histogram[word]['class']] += self.histogram[word]['score']
			y_hat.append(np.argmax(classes))
		return np.array(y_hat)

	def score(self, word, classification):
		"""
		Score function for words.
		:param word: words to score.
		:param classification: Class to score for.
		:return: The score for this word.
		"""
		return (self.histogram[word][classification] - np.mean(self.histogram[word]))**2


def plot_empirical_error(df: DataFrame):
	"""
	An ironically named function (for historical reasons). It actually just prints the empirical 0-1 loss.
	:param df: Data
	"""
	empirical_loss = np.count_nonzero(df['Project'] - df['Prediction']) / len(df['Project'])
	print(1 - empirical_loss)


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
