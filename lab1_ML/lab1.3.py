import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn import datasets


if __name__ == "__main__":
    dataset = pd.read_csv('student_scores.csv')

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(df)

    print("student scores")
    print(f"Coefficient of determination R2:{r2_score(y_test, y_pred)}")
    print(f"Coefficient of determination MAE:{mean_absolute_error(y_test, y_pred)}")
    print(f"Coefficient of determination MAPE:{mean_absolute_percentage_error(y_test, y_pred)}")

    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target
    print("Features:", diabetes.feature_names)

    feature_index = diabetes.feature_names.index('bp')  # Среднее артериальное давление
    X_bp = X[:, np.newaxis, feature_index]

    X_train, X_test, y_train, y_test = train_test_split(X_bp, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    y_approx_sklearn = regressor.predict(X_test)

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_approx_sklearn})
    print(df)

    print("diabets: среднее артериальное давление")
    print(f"Coefficient of determination R2:{r2_score(y_test, y_approx_sklearn)}")
    print(f"Coefficient of determination MAE:{mean_absolute_error(y_test, y_approx_sklearn)}")
    print(f"Coefficient of determination MAPE:{mean_absolute_percentage_error(y_test, y_approx_sklearn)}")