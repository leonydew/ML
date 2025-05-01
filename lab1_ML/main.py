import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def get_csv_data():
	default_file = "student_scores.csv"
	path = input(f"Введите путь к CSV файлу (по умолчанию '{default_file}'): ").strip()
	if path == "":
		path = default_file
	try:
		return pd.read_csv(path)
	except Exception as err:
		print("Ошибка при чтении файла:", err)
		return None


def select_columns(df):
	cols = list(df.columns)
	print("Найденные столбцы:", cols)
	x_col = input("Введите название столбца для X: ").strip()
	y_col = input("Введите название столбца для Y: ").strip()
	if x_col not in df.columns or y_col not in df.columns:
		if len(cols) == 2:
			if x_col == "":
				x_col = cols[0]
			if y_col == "":
				y_col = cols[1]
		else:
			print("Один из указанных столбцов отсутствует в файле.")
			return None, None

	try:
		x = df[x_col].values
		y = df[y_col].values
	except Exception as err:
		print("Один из указанных столбцов отсутствует в файле.", err)
		return None, None
	return x_col, y_col


def print_stats(arr, name):
	print(f"\nСтатистика для столбца '{name}':\n"
		  f"  Количество: {len(arr)}\n"
		  f"  Минимум: {np.min(arr)}\n"
		  f"  Максимум: {np.max(arr)}\n"
		  f"  Среднее: {np.mean(arr)}")


def plot_scatter(x, y, title, x_label, y_label):
	plt.figure(title)
	plt.scatter(x, y, color='blue', label='Данные')
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	plt.legend()


def compute_regression(x, y):
	n = len(x)
	mean_x = np.mean(x)
	mean_y = np.mean(y)
	num = 0
	den = 0
	for xi, yi in zip(x, y):
		num += (xi - mean_x) * (yi - mean_y)
		den += (xi - mean_x) ** 2
	if den == 0:
		print("Ошибка: деление на ноль при вычислении регрессии.")
		return None, None
	return num / den, mean_y - (num / den) * mean_x


def plot_regression_line(x, y, m, b, title, x_label, y_label):
	x_line = np.linspace(np.min(x), np.max(x), 100)
	y_line = m * x_line + b
	plt.figure(title)
	plt.scatter(x, y, color='blue', label='Данные')
	plt.plot(x_line, y_line, color='red', label='Регрессионная прямая')
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	plt.legend()
	return x_line, y_line


def plot_error_squares(x, y, m, b, x_line, y_line, title, x_label, y_label):
	plt.figure(title)
	ax = plt.gca()
	plt.scatter(x, y, color='blue', label='Данные')
	plt.plot(x_line, y_line, color='red', label='Регрессионная прямая')
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	for xi, yi in zip(x, y):
		y_pred = m * xi + b
		error = yi - y_pred
		abs_error = abs(error)
		if abs_error == 0:
			continue
		rect_x = xi - abs_error / 2
		rect_y = y_pred if error >= 0 else yi
		rect = patches.Rectangle(
			(rect_x, rect_y),
			abs_error,
			abs_error,
			linewidth=1,
			edgecolor='green',
			facecolor='green',
			alpha=0.3
		)
		ax.add_patch(rect)
	plt.legend()



# plt.gca().set_aspect('equal', adjustable='box') тут график сжимается по коорд х

def main():
	df = get_csv_data()
	if df is None:
		return

	x_col, y_col = select_columns(df)
	if x_col is None or y_col is None:
		return

	x = df[x_col].values
	y = df[y_col].values

	print_stats(x, x_col)
	print_stats(y, y_col)

	plot_scatter(x, y, "Исходные точки", x_col, y_col)

	m, b = compute_regression(x, y)
	if m is None:
		return

	print(f"\nПараметры регрессионной прямой:\n"
		  f"Наклон (slope) = {m}\n"
		  f"Сдвиг (intercept) = {b}")

	x_line, y_line = plot_regression_line(x, y, m, b, "Регрессионная прямая", x_col, y_col)
	plot_error_squares(x, y, m, b, x_line, y_line, "Квадраты ошибок", x_col, y_col)
	plt.show()


if __name__ == "__main__":
	main()
