import numpy as np
from scipy.optimize import minimize


def main():
    random_state = 0
    n = 100
    np.random.seed(random_state)
    A = np.random.randn(n, n)
    b = np.random.randn(n)
    x_true = A.T @ np.linalg.inv(A @ A.T) @ b

    objective = lambda x: 0.5 * np.linalg.norm(x, ord=2)
    constraints = {"type": "eq", "fun": lambda x: A @ x - b}

    def callback(xk):
        print(f"error: {np.linalg.norm(xk - x_true)}")

    res = minimize(
        fun=objective, x0=np.random.randn(n), constraints=constraints, callback=callback
    )
    assert res.success == np.True_
    assert res.message == "Optimization terminated successfully"
    assert np.allclose(x_true, res.x)


if __name__ == "__main__":
    main()
