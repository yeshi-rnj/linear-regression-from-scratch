"""
Linear Regression from scratch — NumPy only.

Implements:
  - hypothesis function      h(X) = X @ theta
  - MSE cost function        J(theta) = (1/2m) * sum((h(X) - y)^2)
  - gradient of the cost     dJ/dtheta = (1/m) * X^T @ (h(X) - y)
  - batch gradient descent update loop

Works for both simple (single-feature) and multiple (multi-feature)
linear regression — a single feature is just the n=1 case of the
general matrix formulation, so no special-casing is needed.
"""

import numpy as np


class LinearRegressionScratch:
    """Batch gradient descent linear regression, implemented with plain NumPy.

    Parameters
    ----------
    learning_rate : float
        Step size ("alpha") used in each gradient descent update.
    n_iterations : int
        Number of full-batch gradient descent steps to run.
    fit_intercept : bool
        If True, a bias/intercept term is added automatically (a column of
        ones is prepended to X), so the caller does not need to do this
        themselves.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, fit_intercept=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.theta = None          # learned parameters, shape (n_features [+1], 1)
        self.cost_history = []     # cost at every iteration, for the convergence plot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _add_intercept(self, X):
        """Prepend a column of ones to X so theta[0] acts as the intercept."""
        m = X.shape[0]
        ones = np.ones((m, 1))
        return np.hstack([ones, X])

    def _hypothesis(self, X):
        """h(X) = X @ theta  — the model's predictions, as a plain matrix product."""
        return X @ self.theta

    @staticmethod
    def _mse_cost(errors, m):
        """
        Mean-squared-error cost, using the classic 1/(2m) convention so the
        derivative below comes out clean (the 2 cancels the 1/2).

            J(theta) = (1 / 2m) * sum((h(X) - y)^2)
        """
        return float(((errors.T @ errors) / (2 * m)).item())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X, y, verbose=False):
        """
        Train the model with batch gradient descent.

        X : ndarray, shape (m, n)   — m examples, n features
        y : ndarray, shape (m,) or (m, 1)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        if self.fit_intercept:
            X = self._add_intercept(X)

        m, n = X.shape
        self.theta = np.zeros((n, 1))
        self.cost_history = []

        for i in range(self.n_iterations):
            predictions = self._hypothesis(X)          # h(X) = X @ theta
            errors = predictions - y                    # (h(X) - y)

            cost = self._mse_cost(errors, m)
            self.cost_history.append(cost)

            # Gradient of J(theta) w.r.t. theta:
            #   dJ/dtheta = (1/m) * X^T @ (h(X) - y)
            gradient = (X.T @ errors) / m

            # Gradient descent update rule:
            #   theta := theta - alpha * dJ/dtheta
            self.theta = self.theta - self.learning_rate * gradient

            if verbose and (i % max(1, self.n_iterations // 10) == 0):
                print(f"iter {i:5d}  cost = {cost:.6f}")

        return self

    def predict(self, X):
        """Return predictions for new data X, shape (m, n)."""
        X = np.asarray(X, dtype=float)
        if self.fit_intercept:
            X = self._add_intercept(X)
        return (self._hypothesis(X)).ravel()

    @property
    def intercept_(self):
        if not self.fit_intercept:
            return 0.0
        return float(self.theta[0, 0])

    @property
    def coef_(self):
        if self.fit_intercept:
            return self.theta[1:, 0]
        return self.theta[:, 0]


# ----------------------------------------------------------------------
# Manual metrics (no sklearn.metrics allowed inside the implementation)
# ----------------------------------------------------------------------
def mean_squared_error_manual(y_true, y_pred):
    """MSE = (1/m) * sum((y_true - y_pred)^2)"""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score_manual(y_true, y_pred):
    """
    R^2 = 1 - (SS_res / SS_tot)

    SS_res = sum((y_true - y_pred)^2)         residual sum of squares
    SS_tot = sum((y_true - mean(y_true))^2)   total sum of squares
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - (ss_res / ss_tot))
