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
				if word not in self.histogram:
					self.histogram[word] = [0 for i in range(7)]
				self.histogram[word][class_of_line] += 1*weight_of_line

	def predict(self, X):
		y_hat = []
		for sample in tqdm(X, desc="Predicting"):
			classes = [0 for i in range(7)]
			for word in sample.split():
				if word in self.histogram:
					word_class = np.argmax(self.histogram[word])
					classes[word_class] += 1
			y_hat.append(np.argmax(classes))
		return np.array(y_hat)


def plot_empirical_error(df: DataFrame):
	empirical_loss = np.count_nonzero(df['Project'] - df['Prediction']) / len(df['Project'])
	print(empirical_loss)


if __name__ == '__main__':
	model = BaseLine()
	train, val, eval = get_train_validate_evaluate()
	model.train(train)
	print(len(model.histogram))
	y_hat = model.predict(train['line'])
	train['Prediction'] = y_hat
	plot_empirical_error(train)
	y_hat = model.predict(val['line'])
	val['Prediction'] = y_hat
	plot_empirical_error(val)
