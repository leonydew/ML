import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target
    print("Features:", diabetes.feature_names)

    feature_index = diabetes.feature_names.index('bp') #Среднее артериальное давление
    X_bp = X[:, np.newaxis, feature_index]

    X_train, X_test, y_train, y_test = train_test_split(X_bp, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    y_approx_sklearn = regressor.predict(X_test)

    print(f"Scikit-Learn коэффициенты: w1 = {regressor.coef_}, w0 = {regressor.intercept_}")

    x = X_bp.flatten()
    w0 = calc_w0(X_train, y_train)
    w1 = calc_w1(X_train, y_train)
    y_approx = []
    for i in X_test:
        y_approx.append(f(i, w0, w1))

    print(f"мои коэффициенты: w1 = {w1}, w0 = {w0}")

    plt.scatter(x, y, color="blue")
    plt.plot(X_test, y_approx_sklearn, color="black")
    plt.plot(X_test, y_approx, color="yellow", linestyle=":")
    plt.xlabel("Среднее артериальное давление")
    plt.ylabel("Диабет")
    plt.title("Линейная регрессия")
    plt.grid(True)
    plt.show()

    df = pd.DataFrame({"Actual": y_test, "sklearn": y_approx_sklearn, "my predict": y_approx})
    print(df)