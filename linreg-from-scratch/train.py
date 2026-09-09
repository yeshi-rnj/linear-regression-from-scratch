"""
Train and evaluate the from-scratch LinearRegressionScratch model, then
sanity-check it against scikit-learn's LinearRegression.

Dataset: sklearn's built-in "diabetes" dataset (442 patients, 10 numeric
baseline measurements as features, disease-progression-after-one-year as
the continuous target). It's a real dataset and ships with scikit-learn,
so it doesn't require any network access.

Produces:
  - plots/cost_convergence_multi.png   (multi-feature cost vs. iteration)
  - plots/cost_convergence_simple.png  (single-feature cost vs. iteration)
  - plots/predicted_vs_actual.png
  - results.json                       (all numbers used in the README)
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from linear_regression import (
    LinearRegressionScratch,
    mean_squared_error_manual,
    r2_score_manual,
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def standardize(train, test):
    """Standardize using train-set statistics only (avoids test-set leakage)."""
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    return (train - mean) / std, (test - mean) / std, mean, std


def main():
    # ------------------------------------------------------------------
    # 1. Load data with pandas, check for missing values
    # ------------------------------------------------------------------
    data = load_diabetes(as_frame=True)
    df = data.frame
    print("Dataset shape:", df.shape)
    print("Missing values per column:\n", df.isna().sum())

    feature_cols = list(data.feature_names)
    target_col = "target"

    X = df[feature_cols].values
    y = df[target_col].values

    # ------------------------------------------------------------------
    # 2. Train/test split (sklearn's splitter is fine to use — only the
    #    modeling itself has to be hand-built)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Standardize features (fit on train only)
    X_train_s, X_test_s, mean, std = standardize(X_train, X_test)

    results = {}

    # ==================================================================
    # PART A — Multiple linear regression (all 10 features)
    # ==================================================================
    model = LinearRegressionScratch(learning_rate=0.5, n_iterations=2000)
    model.fit(X_train_s, y_train, verbose=True)

    train_pred = model.predict(X_train_s)
    test_pred = model.predict(X_test_s)

    mse_train = mean_squared_error_manual(y_train, train_pred)
    mse_test = mean_squared_error_manual(y_test, test_pred)
    r2_train = r2_score_manual(y_train, train_pred)
    r2_test = r2_score_manual(y_test, test_pred)

    print(f"\n[Scratch - multi-feature] train MSE={mse_train:.3f} R2={r2_train:.4f}")
    print(f"[Scratch - multi-feature] test  MSE={mse_test:.3f} R2={r2_test:.4f}")

    # Cost convergence plot
    plt.figure(figsize=(7, 5))
    plt.plot(model.cost_history)
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(theta)  (MSE / 2)")
    plt.title("Gradient Descent Convergence — Multiple Linear Regression")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/cost_convergence_multi.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # scikit-learn comparison (multi-feature)
    # ------------------------------------------------------------------
    sk_model = LinearRegression()
    sk_model.fit(X_train_s, y_train)
    sk_test_pred = sk_model.predict(X_test_s)

    sk_mse_test = mean_squared_error_manual(y_test, sk_test_pred)  # manual metric, sklearn model
    sk_r2_test = r2_score_manual(y_test, sk_test_pred)

    coef_compare = pd.DataFrame(
        {
            "feature": feature_cols,
            "scratch_coef": model.coef_,
            "sklearn_coef": sk_model.coef_,
            "abs_diff": np.abs(model.coef_ - sk_model.coef_),
        }
    )
    print("\nCoefficient comparison (multi-feature):")
    print(coef_compare.to_string(index=False))
    print(f"\nIntercept  scratch={model.intercept_:.4f}  sklearn={sk_model.intercept_:.4f}")
    print(f"Test R2    scratch={r2_test:.4f}  sklearn={sk_r2_test:.4f}")
    print(f"Test MSE   scratch={mse_test:.4f}  sklearn={sk_mse_test:.4f}")

    results["multi_feature"] = {
        "features_used": feature_cols,
        "learning_rate": model.learning_rate,
        "n_iterations": model.n_iterations,
        "scratch": {
            "train_mse": mse_train,
            "test_mse": mse_test,
            "train_r2": r2_train,
            "test_r2": r2_test,
            "intercept": model.intercept_,
            "coefficients": dict(zip(feature_cols, model.coef_.tolist())),
        },
        "sklearn": {
            "test_mse": sk_mse_test,
            "test_r2": sk_r2_test,
            "intercept": float(sk_model.intercept_),
            "coefficients": dict(zip(feature_cols, sk_model.coef_.tolist())),
        },
        "max_abs_coefficient_diff": float(coef_compare["abs_diff"].max()),
    }

    # Predicted vs actual scatter (test set)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, test_pred, alpha=0.6, label="Scratch model", edgecolor="k", linewidth=0.3)
    lims = [min(y_test.min(), test_pred.min()), max(y_test.max(), test_pred.max())]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual target")
    plt.ylabel("Predicted target")
    plt.title("Predicted vs. Actual — Test Set (Multi-feature)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/predicted_vs_actual.png", dpi=150)
    plt.close()

    # ==================================================================
    # PART B — Simple linear regression (single feature: BMI)
    # BMI ("bmi") is the single strongest predictor of disease progression
    # in this dataset, so it's used to demonstrate the single-feature case.
    # ==================================================================
    bmi_idx = feature_cols.index("bmi")
    X_train_simple = X_train_s[:, [bmi_idx]]
    X_test_simple = X_test_s[:, [bmi_idx]]

    simple_model = LinearRegressionScratch(learning_rate=0.5, n_iterations=2000)
    simple_model.fit(X_train_simple, y_train)

    simple_test_pred = simple_model.predict(X_test_simple)
    simple_mse_test = mean_squared_error_manual(y_test, simple_test_pred)
    simple_r2_test = r2_score_manual(y_test, simple_test_pred)

    sk_simple = LinearRegression()
    sk_simple.fit(X_train_simple, y_train)
    sk_simple_pred = sk_simple.predict(X_test_simple)
    sk_simple_mse = mean_squared_error_manual(y_test, sk_simple_pred)
    sk_simple_r2 = r2_score_manual(y_test, sk_simple_pred)

    print(f"\n[Scratch - simple/BMI] test MSE={simple_mse_test:.3f} R2={simple_r2_test:.4f}")
    print(f"[sklearn - simple/BMI] test MSE={sk_simple_mse:.3f} R2={sk_simple_r2:.4f}")
    print(
        f"Coef  scratch={simple_model.coef_[0]:.4f}  sklearn={sk_simple.coef_[0]:.4f}  "
        f"| Intercept scratch={simple_model.intercept_:.4f}  sklearn={sk_simple.intercept_:.4f}"
    )

    plt.figure(figsize=(7, 5))
    plt.plot(simple_model.cost_history)
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(theta)  (MSE / 2)")
    plt.title("Gradient Descent Convergence — Simple Linear Regression (BMI only)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/cost_convergence_simple.png", dpi=150)
    plt.close()

    results["simple_feature"] = {
        "feature_used": "bmi",
        "scratch": {
            "test_mse": simple_mse_test,
            "test_r2": simple_r2_test,
            "coefficient": float(simple_model.coef_[0]),
            "intercept": simple_model.intercept_,
        },
        "sklearn": {
            "test_mse": sk_simple_mse,
            "test_r2": sk_simple_r2,
            "coefficient": float(sk_simple.coef_[0]),
            "intercept": float(sk_simple.intercept_),
        },
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved plots/ and results.json")


if __name__ == "__main__":
    main()
