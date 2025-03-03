# https://www.kaggle.com/datasets/oktayrdeki/ddos-traffic-dataset

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, recall_score, 
	precision_score, f1_score)
from tabulate import tabulate

def uploading_and_checking_data(file_path: str):
	# Загрузка данных, а также их проверка на пропущенные значения и предпросмотр
	df = pd.read_csv(file_path)

	print("\nПропущенные значения:")
	print(df.isnull().sum())

	print("\nПредпросмотр данных:")
	print(tabulate(df.head(5), headers=df.columns, tablefmt='grid'))

	return df

def preprocessing_data(df: pd.DataFrame):
	l_encoder = LabelEncoder()

	# Кодирование категориальных признаков
	df['Highest Layer'] = l_encoder.fit_transform(df['Highest Layer'])
	df['Transport Layer'] = l_encoder.fit_transform(df['Transport Layer'])

	# Разделение на признаки и целевую переменную
	X = df[['Highest Layer', 'Transport Layer', 'Packet Length', 'Packets/Time']]
	y = df['target']

	return X, y

def evaluate_model(y_test: pd.Series, y_pred: pd.Series):
	# Производим оценку качества модели
	metrics = {
		"Accuracy": accuracy_score(y_test, y_pred),
		"Recall": recall_score(y_test, y_pred),
		"Precision": precision_score(y_test, y_pred),
		"F1-Score": f1_score(y_test, y_pred)
	}

	# Вывод метрик
	print("Метрики оценки модели:")
	for metric_name, metric_value in metrics.items():
		print(f"{metric_name}: {metric_value:.4f}")

def main(file_path: str):
	# Загрузка данных
	df = uploading_and_checking_data(file_path)

	# Предобработка данных
	X, y = preprocessing_data(df)

	# Разделение данных на тренировачные и тестовые
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Обучени модели
	model = DecisionTreeClassifier()
	model.fit(X_train, y_train)

	# Оценка модели
	y_pred = model.predict(X_test)
	evaluate_model(y_test, y_pred)

if __name__ == '__main__':
	main('DDoS_dataset.csv')