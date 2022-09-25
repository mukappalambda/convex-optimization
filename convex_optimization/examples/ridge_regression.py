import cvxpy as cp
import numpy as np
from sklearn.linear_model import Ridge


def objective_fn(x: np.ndarray, y: np.ndarray, w, alpha):
    return cp.pnorm(cp.matmul(x, w) - y, p=2) ** 2 + alpha * cp.pnorm(w, p=2) ** 2


def main():
    n_samples = 500
    n_features = 4
    alpha_true = 0.4321
    random_state = 0
    np.random.seed(random_state)
    x = np.random.randn(n_samples, n_features)
    w_true = np.array([1, 2, 3, 4])
    y = np.matmul(x, w_true)

    # solve ridge regression using cvxpy
    w = cp.Variable(shape=n_features)
    alpha = cp.Parameter(nonneg=True)
    problem = cp.Problem(
        objective=cp.Minimize(objective_fn(x=x, y=y, w=w, alpha=alpha))
    )
    alpha.value = alpha_true
    problem.solve()
    assert problem.status == "optimal"

    # solve ridge regression using sklearn
    ridge = Ridge(alpha=alpha_true, solver="svd", random_state=random_state)
    ridge.fit(x, y)

    MESSAGE = f"""
    true solution: {w_true}
    cvxpy's solution: {w.value}
    klearn's solution: {ridge.coef_}
    """
    print(MESSAGE)


if __name__ == "__main__":
    main()
