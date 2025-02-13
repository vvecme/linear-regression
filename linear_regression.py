""" Программа для регрессионого анализа данных.
Формат файла - txt.
Вид:
Первая строка - наименования переменных (0 - это отклик)
Первый столбец - номер строки
2, 3, 4 столбцы - значения трех переменных (их имена "1", "5" и "3")
5 столбец отклик, зависимая переменная (поименована как "0")
"""

import numpy as np
import os
import sklearn.linear_model as sklin


ROUND = 4 # до какого знака округлять

filename = 'Tabl.txt'

# директория по умолчанию: нахождение скрипта
py_file_dir = os.path.dirname(os.path.abspath(__file__))
work_folder = os.chdir(py_file_dir)

# Проверка существования файла
if not os.path.isfile(FILENAME):
    raise FileNotFoundError(f"Файл {FILENAME} не найден.")

# Загрузка данных
try:
    columns = np.genfromtxt(FILENAME, skip_header=1).T
except Exception as e:
    raise RuntimeError(f"Ошибка при чтении файла: {e}")

columns = np.genfromtxt(filename, skip_header=1).T
X = columns[1:4].copy().T # независимые переменные
Y = columns[4].copy() # зависимая переменная

model = sklin.LinearRegression() # простые наименьшие квадраты
model.fit(x, y)

coefficients = model.coef_
intercept = model.intercept_
error = np.max(np.fabs(np.subtract(y, model.predict(x))))

print(f'Линейная модель: y = {round(intercept, ROUND)} '
        f'+ {round(coefficients[0], ROUND)}x1 '
        f'+ {round(coefficients[1], ROUND)}x2 '
        f'+ {round(coefficients[2], ROUND)}x3. Ошибка - {round(error, ROUND)}.')


