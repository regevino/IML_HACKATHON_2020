from typing import Tuple
import numpy as np
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ALL_FILES = ["building_tool_all_data.txt", "espnet_all_data.txt", "horovod_all_data.txt",
			 "jina_all_data.txt", "PaddleHub_all_data.txt", "PySolFC_all_data.txt",
			 "pytorch_geometric_all_data.txt"]

LABELS = {"building_tool_all_data.txt": 0, "espnet_all_data.txt": 1, "horovod_all_data.txt": 2,
		  "jina_all_data.txt": 3, "PaddleHub_all_data.txt": 4, "PySolFC_all_data.txt": 5,
		  "pytorch_geometric_all_data.txt": 6}


def get_all_data():
	lines = []
	cls = []
	for file in tqdm(ALL_FILES, desc="Reading files..."):
		with open(file, encoding='utf-8') as f:
			file_lines = f.readlines()
			file_lines = list(filter(line_cleaner, file_lines))
			lines.extend(file_lines)
			class_num = LABELS[file]

			cls.extend([class_num for i in range(len(file_lines))])

	return DataFrame({'line': lines, 'Project': cls})


def line_cleaner(line: str):
	if not line.strip():
		return False
	return True


def split(df: DataFrame):
	train, rest = train_test_split(df, train_size=0.6, random_state=42)
	validate, evaluate = train_test_split(rest, train_size=0.5, random_state=42)
	return train, validate, evaluate


def get_train_validate_evaluate() -> Tuple[DataFrame, DataFrame, DataFrame]:
	return split(get_all_data())


def feature_creation(samples: np.ndarray, histogram: dict):
	X = []
	for row in tqdm(samples, desc="Creating Features"):
		sample = row[0]
		classes = [0 for i in range(7)]
		for word in sample.split():
			word = word.strip()
			if not word:
				break
			if word in histogram:
				# word_class = np.argmax(self.histogram[word])
				classes[histogram[word]['class']] += histogram[word]['score']
		X.append(classes)
	# res_df = DataFrame({'X': X, 'y': df['Project'].values})
	return X


def create_histogram(df: DataFrame):
	histogram = dict()
	for row in tqdm(df.values, desc="Creating Histogram"):
		line_of_code = row[0]
		class_of_line = row[1]
		for word in line_of_code.split():
			word = word.strip()
			if not word:
				break
			if word not in histogram:
				histogram[word] = [0 for i in range(7)]
			histogram[word][class_of_line] += 1

	for word in tqdm(histogram, desc="Normalizing"):
		histogram[word] = (np.array(histogram[word]) - np.min(histogram[word])) / np.max(
			histogram[word])
		classification = np.argmax(histogram[word])
		_score = score(histogram, word, classification)
		histogram[word] = dict()
		histogram[word]['class'] = classification
		histogram[word]['score'] = _score
	return histogram


def score(histogram, word, classification):
	return (histogram[word][classification] - np.mean(histogram[word])) ** 2


if __name__ == '__main__':
	train, validate, eval = get_train_validate_evaluate()
	histogram = create_histogram(train)

	train_X = feature_creation(train.values, histogram)
	val_X = feature_creation(validate.values, histogram)

	eval_X = feature_creation(eval.values, histogram)

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
	# # print(model.score(train_X, train_y))
	# # # type.append('train')
	# # print(model.score(val_X, val_y))
	# print(model.score(eval_X, eval_y))

# type.append('validate')
# depths_k.append(depth)
# depths_k.append(depth)

# plot = ggplot(DataFrame({"iterations": depths_k, 'score': score, 'type': type}))
# plot += geom_line(aes(x='iterations', y='score', linetype='type'))
# ggsave(plot, 'Adaboost depth 2.png')
# print(plot)
