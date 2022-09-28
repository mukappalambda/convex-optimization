import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    test_size = 0.2
    random_state = 0
    max_iters = 100
    x, y = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    scaled_train: np.ndarray = scaler.fit_transform(x_train)
    scaled_test: np.ndarray = scaler.transform(x_test)

    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    beta = np.random.randn(x_train.shape[1])

    lr = 1e-3
    for i in range(max_iters):
        mu = sigmoid(scaled_train @ beta)
        g = scaled_train.T @ (mu - y_train)
        beta -= lr * g
        if i % 10 == 0:
            i = i * 0.9

    train_preds = (sigmoid(scaled_train @ beta) > 0.5).astype(int)
    test_preds = (sigmoid(scaled_test @ beta) > 0.5).astype(int)
    print(f"gradient descent train score: {accuracy_score(train_preds, y_train)}")
    print(f"gradient descent test score: {accuracy_score(test_preds, y_test)}")


if __name__ == "__main__":
    main()
