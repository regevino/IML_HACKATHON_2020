from typing import Tuple

from pandas import DataFrame
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


def feature_creation(code_lines):
	words = set()
	for row in tqdm(code_lines, desc="Finding Features"):
		for word in row[0].split():
			words.add(word)
	words_list = list(words)
	X = [words]
	word_indices = {word: index for index, word in
					enumerate(tqdm(words_list, desc="Creating Feature indices"))}
	linse = dict()
	for sample in tqdm(code_lines, desc="Creating Sample Matrix"):
		line = sample[0]
		row = {word: 0 for word in words}
		for word in line.split():
			if word in words:
				row[word] += 1
		linse[line] = row

	return DataFrame(linse)


## THIS WAS BAD ##


if __name__ == '__main__':
	train, validate, eval = get_train_validate_evaluate()
	print(train)
