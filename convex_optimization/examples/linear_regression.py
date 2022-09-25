import cvxpy as cp
import numpy as np


def generate_data(n_samples: int, random_state: int = 0):
    np.random.seed(random_state)
    w_true = np.array([1, 2]).reshape(-1, 1)
    x = np.random.randn(n_samples, 1)
    noise = 1e-3 * np.random.randn(n_samples, 1)
    y: np.ndarray = w_true[0] + x * w_true[1:] + noise
    assert x.shape == (n_samples, 1)
    assert x.shape == y.shape
    assert x.shape[1] + 1 == w_true.shape[0]
    return x, y, w_true


def objective_fn(x: np.ndarray, y: np.ndarray, w):
    A = np.hstack([np.ones_like(x), x])
    return cp.sum_squares(cp.matmul(A, w) - y)


def main():
    n_samples = 1000
    random_state = 0
    x, y, w_true = generate_data(n_samples=n_samples, random_state=random_state)

    w_hat = cp.Variable(shape=w_true.shape)
    problem = cp.Problem(objective=cp.Minimize(objective_fn(x=x, y=y, w=w_hat)))
    problem.solve()
    assert problem.status == "optimal"
    assert np.allclose(w_true, w_hat.value, atol=1e-3)

    print(f"w_true: {w_true.ravel()}; w_hat: {w_hat.value.ravel()}")


if __name__ == "__main__":
    main()
