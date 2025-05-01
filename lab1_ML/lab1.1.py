import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pathlib


def calc_w1(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(x[i] * y[i] for i in range(n))
    sum_x2 = sum(x)**2
    sum_xx = sum(x[i]**2 for i in range(n))

    return (sum_x * sum_y / n - sum_xy)/(sum_x2 / n - sum_xx)


def calc_w0(x, y):
    sum_x = sum(x)
    sum_y = sum(y)
    n = len(x)
    return sum_y / n - calc_w1(x, y) * sum_x / n


def f(x, w0, w1):
    return w0 + w1 * x

if __name__ == "__main__":
    filename = input('enter your path to .csv file:')
    df = pd.read_csv(pathlib.Path(filename))

    col1_name = df.columns[0]
    col2_name = df.columns[1]
    choose = int(input(f"choose axis:\n1)x:{col1_name} y:{col2_name}\n2)x:{col2_name} y:{col1_name}"))

    if(choose == 1):
        values1 = df[col1_name]
        values2 = df[col2_name]
    elif(choose == 2):
        values1 = df[col2_name]
        values2 = df[col1_name]
        col1_name = df.columns[1]
        col2_name = df.columns[0]
    else:
        print("error! Wrong number")

    num1 = df.count().values[0]
    num2 = df.count().values[1]

    print(f"количество по каждому из столбцов:\n{col1_name}: {num1}\n"
          f"{col2_name}: {num2}")

    print(f"min по каждому из столбцов:\n{col1_name}: {min(values1)}\n"
          f"{col2_name}: {min(values2)}")

    print(f"max по каждому из столбцов:\n{col1_name}: {max(values1)}\n"
          f"{col2_name}: {max(values2)}")

    print(f"средне по каждому из столбцов:\n{col1_name}: {sum(values1)/num1}\n"
          f"{col2_name}: {sum(values2)/num2}")

    w0 = calc_w0(values1, values2)
    w1 = calc_w1(values1, values2)
    y_approx = []
    for i in values1:
        y_approx.append(f(i, w0, w1))

    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 2)

    grid1 = fig.add_subplot(grid[0, 0])
    grid1.scatter(values1, values2, linewidths=0.5)
    grid1.set_title("task3")
    grid1.set_xlabel(col1_name)
    grid1.set_ylabel(col2_name)

    grid2 = fig.add_subplot(grid[0, 1])
    grid2.scatter(df[col1_name], df[col2_name], linewidths=0.5)
    grid2.plot(values1, y_approx, color='red')
    grid2.set_title("task5")
    grid2.set_xlabel(col1_name)
    grid2.set_ylabel(col2_name)

    grid3 = fig.add_subplot(grid[1, 0])
    grid3.scatter(df[col1_name], df[col2_name], linewidths=0.5)
    grid3.plot(values1, y_approx, color='red')
    for i in range(len(values1)):
        width = y_approx[i] - values2[i]
        rect = Rectangle((values1[i], values2[i]), width, width, linewidth=1, edgecolor='g', facecolor='none', fill=None, hatch='//')
        grid3.add_patch(rect)
    grid3.set_title("task6")
    grid3.set_xlabel(col1_name)
    grid3.set_ylabel(col2_name)

    plt.show()