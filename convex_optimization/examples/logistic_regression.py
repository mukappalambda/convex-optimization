import cvxpy as cp
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    test_size = 0.2
    random_state = 0
    x, y = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(x_train)
    lr = LogisticRegression(random_state=random_state)
    lr.fit(scaled_train, y_train)
    print(f"sklearn train score: {lr.score(scaled_train, y_train)}")
    scaled_test = scaler.transform(x_test)
    print(f"sklearn test score: {lr.score(scaled_test, y_test)}")

    beta = cp.Variable(scaled_train.shape[1])
    lambd = cp.Parameter(nonneg=True)
    log_likelihood = cp.sum(
        cp.multiply(y_train, scaled_train @ beta) - cp.logistic(scaled_train @ beta)
    )
    problem = cp.Problem(
        cp.Minimize(
            -log_likelihood / scaled_train.shape[0] + lambd * cp.pnorm(beta, p=2)
        )
    )
    lambd.value = 1
    problem.solve()
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    train_preds = (sigmoid(scaled_train @ beta.value) > 0.5).astype(int)
    test_preds = (sigmoid(scaled_test @ beta.value) > 0.5).astype(int)
    assert train_preds.shape == y_train.shape
    assert test_preds.shape == y_test.shape
    print(f"cvxpy train score: {accuracy_score(train_preds, y_train)}")
    print(f"cvxpy test score: {accuracy_score(test_preds, y_test)}")


if __name__ == "__main__":
    main()
