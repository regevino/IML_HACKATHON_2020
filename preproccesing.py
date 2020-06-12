from typing import Tuple
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_LABEL = 'Project'

CODE_LINES = 'line'

ALL_FILES = ["building_tool_all_data.txt", "espnet_all_data.txt", "horovod_all_data.txt",
			 "jina_all_data.txt", "PaddleHub_all_data.txt", "PySolFC_all_data.txt",
			 "pytorch_geometric_all_data.txt"]

LABELS = {"building_tool_all_data.txt": 0, "espnet_all_data.txt": 1, "horovod_all_data.txt": 2,
		  "jina_all_data.txt": 3, "PaddleHub_all_data.txt": 4, "PySolFC_all_data.txt": 5,
		  "pytorch_geometric_all_data.txt": 6}


def get_all_data() -> DataFrame:
	"""
	Read the github-project files and load them line by line into a pandas DataFrame along with their class.
	:return: pandas DataFrame with CODE_LINES as the key for lines of code and PROJET_LABEL as the key for
	their class.
	"""
	lines = []
	cls = []
	for file in tqdm(ALL_FILES, desc="Reading files..."):
		with open(file, encoding='utf-8') as f:
			file_lines = f.readlines()
			file_lines = list(filter(line_cleaner, file_lines))
			lines.extend(file_lines)
			class_num = LABELS[file]
			cls.extend([class_num for i in range(len(file_lines))])
	return DataFrame({CODE_LINES: lines, PROJECT_LABEL: cls})


def line_cleaner(line: str) -> bool:
	"""
	Small helper function for filtering out empty lines etc.
	Further tinkering here could provide better score for the learner as there are many lines that have
	close to no information (such as empty lines, or '{').
	:param line: A line of code.
	:return: True if is should be kept, False otherwise.
	"""
	if not line.strip():
		return False
	return True


def split(df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
	"""
	Split the Data three-ways, so that .6 of it is for training, .2 for validation and .2 for evaluation.
	:param df: pandas DataFrame containing the data.
	:return: A tuple of three DataFrames, for Train, Validate, Evaluate respectively.
	"""
	tr, rest = train_test_split(df, train_size=0.6, random_state=42)
	val, evaluate = train_test_split(rest, train_size=0.5, random_state=42)
	return tr, val, evaluate


def get_train_validate_evaluate() -> Tuple[DataFrame, DataFrame, DataFrame]:
	"""
	Get the split data set, ready for learning.
	:return: A tuple of three DataFrames, for Train, Validate, Evaluate respectively.
	"""
	return split(get_all_data())


def feature_creation(samples: np.ndarray, histogram: dict) -> list:
	"""
	Create features for every sample, by summing the score for each word in the sample.
	:param samples: Numpy array containing lines of code as samples.
	:param histogram: A histogram creating the score and class calculated for each word.
	:return: A m*7 matrix, where m is the number of lines.
	"""
	X = []
	for row in tqdm(samples, desc="Creating Features"):
		sample = row
		classes = [0 for i in range(7)]
		for word in sample.split():
			word = word.strip()
			if not word:
				break
			if word in histogram:
				classes[histogram[word]['class']] += histogram[word]['score']
		X.append(classes)
	return X


def create_histogram(df: DataFrame) -> dict:
	"""
	Create a histogram for all the words in the training data, normalizing and scoring each word for the
	class it 'belongs to'
	:param df: panda DataFrame containing the the training data.
	:return: A dictionary containing the histogram data.
	"""
	hist = dict()
	for row in tqdm(df.values, desc="Creating Histogram"):
		line_of_code = row[0]
		class_of_line = row[1]
		for word in line_of_code.split():
			word = word.strip()
			if not word:
				break
			if word not in hist:
				hist[word] = [0 for i in range(7)]
			hist[word][class_of_line] += 1

	for word in tqdm(hist, desc="Normalizing"):
		hist[word] = (np.array(hist[word]) - np.min(hist[word])) / np.max(
			hist[word])
		classification = np.argmax(hist[word])
		_score = score(hist, word, classification)
		hist[word] = dict()
		hist[word]['class'] = classification
		hist[word]['score'] = _score
	return hist


def score(hist: dict, word: str, classification: int) -> float:
	"""
	Score the word for this class.
	:param hist: dictionary containing histogram data.
	:param word: Word to score
	:param classification: Class to score this word for.
	:return: The score this word gets for this class.
	"""
	return (hist[word][classification] - np.mean(hist[word])) ** 2

# Code we used for cross validation and model selection:

# if __name__ == '__main__':
# 	train, validate, eval = get_train_validate_evaluate()
# 	histogram = create_histogram(train)
#
# 	train_X = feature_creation(train['line'].values, histogram)
# 	train_y = train['Project'].values
# 	val_X = feature_creation(validate['line'].values, histogram)
# 	val_y = validate['Project'].values
#
# 	eval_X = feature_creation(eval['line'].values, histogram)

# alphas = [1, 5, 20, 30, 70, 100]
# score = []
# type = []
# depths_k = []
# for alpha in tqdm(alphas, desc="Adaboost for iterations"):
# 	model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=alpha)
# 	model.fit(train_X, train_y)
# 	score.append(model.score(train_X, train_y))
# 	type.append('train')
# 	score.append(model.score(val_X, val_y))
# 	type.append('validate')
# 	depths_k.append(alpha)
# 	depths_k.append(alpha)

# model = RandomForestClassifier(max_depth=25)
# model.fit(train_X, train_y)
# print(model.score(train_X, train_y))
# # # type.append('train')
# print(model.score(val_X, val_y))
# print(model.score(eval_X, eval_y))

# type.append('validate')
# depths_k.append(depth)
# depths_k.append(depth)

# plot = ggplot(DataFrame({"iterations": depths_k, 'score': score, 'type': type}))
# plot += geom_line(aes(x='iterations', y='score', linetype='type'))
# ggsave(plot, 'Adaboost depth 2.png')
# print(plot)
