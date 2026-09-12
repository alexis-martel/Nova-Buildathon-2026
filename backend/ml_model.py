# First I want to test regression models
# However, I want to test them for all of my features
# (mean, median, or Q75 of my epochs per electrode)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score, roc_curve)
from sklearn.model_selection import LeaveOneGroupOut
import joblib

def run_regression_models():
    """
    Run linear and quadtratic regression independently on each EEG metric. 

    """

    # Columns that are not EEG metrics
    df = pd.read_csv(r'n_back_theta_welch_relative_summary.csv')
    exclude_cols = ["Sample", "Condition"]

    metrics = [col for col in df.columns if col not in exclude_cols]

    # Convert N-back condition to ordered numerical values
    condition_mapping = {"1-back": 1, "2-back": 2, "3-back": 3,"4-back": 4}

    data = df.copy()
    data["Workload"] = data["Condition"].map(condition_mapping)

    if data["Workload"].isna().any():
        raise ValueError("Unknown condition found in Condition column.")

    # Leave-one-sample-out cross-validation
    logo = LeaveOneGroupOut()

    results = []

    for metric in metrics:

        metric_data = data[["Sample", "Condition", "Workload", metric]].dropna()

        X = metric_data[[metric]]
        y = metric_data["Workload"]
        groups = metric_data["Sample"]

        # Linear Regression

        linear_true = []
        linear_pred = []

        for train_idx, test_idx in logo.split(X, y, groups):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            model = LinearRegression()
            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            linear_true.extend(y_test)
            linear_pred.extend(pred)

        results.append({
            "Metric": metric,
            "Model": "Linear",
            "RMSE": np.sqrt(
                mean_squared_error(linear_true, linear_pred)
            ),
            "MAE": mean_absolute_error(
                linear_true, linear_pred
            ),
            "R2": r2_score(
                linear_true, linear_pred
            )
        })

        quadratic_true = []
        quadratic_pred = []
        for train_idx, test_idx in logo.split(X, y, groups):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            model = make_pipeline(
                PolynomialFeatures(degree=2),
                LinearRegression()
            )

            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            quadratic_true.extend(y_test)
            quadratic_pred.extend(pred)

        results.append({
            "Metric": metric,
            "Model": "Quadratic",
            "RMSE": np.sqrt(
                mean_squared_error(
                    quadratic_true,
                    quadratic_pred
                )
            ),
            "MAE": mean_absolute_error(
                quadratic_true,
                quadratic_pred
            ),
            "R2": r2_score(
                quadratic_true,
                quadratic_pred
            )
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv('n_back_theta_welch_relative_regression_results.csv')


def run_logistic_regression_models():
    """
    Run logistic regression separately for each EEG feature.

    1-back is encoded as 0 and 4-back as 1.
    2-back and 3-back observations are dropped.

    Leave-One-Participant-Out cross-validation is used so that
    data from the same participant is never present in both
    the training and testing sets.

    Returns a DataFrame containing classification scores
    for each feature.
    """
    df = pd.read_csv('n_back_theta_welch_summary.csv')
    exclude_cols = ['Sample', 'Condition']
    features = [col for col in df.columns if col not in exclude_cols]

    # Keep only 1-back and 4-back conditions
    data = df[df["Condition"].isin(["1-back", "4-back"])].copy()

    # Encode 1-back as 0 and 4-back as 1
    data["Workload"] = data["Condition"].map({
        "1-back": 0,
        "4-back": 1
    })

    logo = LeaveOneGroupOut()

    results = []

    for feature in features:

        # Remove rows where this feature is missing
        feature_data = data[
            ["Sample", "Workload", feature]
        ].dropna()

        X = feature_data[[feature]]
        y = feature_data["Workload"]
        groups = feature_data["Sample"]

        y_true = []
        y_pred = []
        y_prob = []

        # Leave one participant out at a time
        for train_idx, test_idx in logo.split(X, y, groups):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            model = LogisticRegression()
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]

            y_true.extend(y_test)
            y_pred.extend(predictions)
            y_prob.extend(probabilities)

        # Calculate classification scores
        results.append({
            "Feature": feature,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_true, y_prob)
        })
    pd.DataFrame(results).to_csv('logit_regression_absolute_theta_welch_results.csv')
    return pd.DataFrame(results)


def run_multiclass_logistic_regression():
    """
    Run multiclass logistic regression separately for each EEG feature.

    Conditions are encoded as:
        1-back -> 0
        2-back -> 1
        3-back -> 2
        4-back -> 3

    Leave-One-Participant-Out cross-validation is used so that
    data from the same participant is never present in both
    the training and testing sets.

    Returns a DataFrame containing classification scores
    for each feature.
    """
    df = pd.read_csv(r'n_back_theta_welch_relative_summary.csv')

    features = [
        "F3",
        "F4",
        "F7",
        "F8",
        "Fz",
        "Frontal_Mean_Relative"
    ]

    # Encode the N-back conditions
    data = df.copy()

    data["Workload"] = data["Condition"].map({
        "1-back": 0,
        "2-back": 1,
        "3-back": 2,
        "4-back": 3
    })

    # Remove any conditions that were not recognised
    data = data.dropna(subset=["Workload"])

    logo = LeaveOneGroupOut()

    results = []

    for feature in features:

        # Remove rows where this feature is missing
        feature_data = data[
            ["Sample", "Workload", feature]
        ].dropna()

        X = feature_data[[feature]]
        y = feature_data["Workload"]
        groups = feature_data["Sample"]

        y_true = []
        y_pred = []

        # Leave one participant out at a time
        for train_idx, test_idx in logo.split(X, y, groups):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            model = LogisticRegression(
                max_iter=1000
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            y_true.extend(y_test)
            y_pred.extend(predictions)

        # Calculate classification scores
        results.append({
            "Feature": feature,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced_Accuracy": balanced_accuracy_score(
                y_true,
                y_pred
            ),
            "Precision_Macro": precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),
            "Recall_Macro": recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),
            "F1_Macro": f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        })
    pd.DataFrame(results).to_csv('logit_regression_all_n_backs_welch.csv')


def best_logistic_regression_model():

    df = pd.read_csv('n_back_theta_welch_relative_summary.csv')

    # Keep only the variables we need
    df = df[
        ["Sample", "Condition", "Fz"]
    ].copy()

    # Keep only 1-back and 4-back
    data = df[
        df["Condition"].isin(["1-back", "4-back"])
    ].copy()

    # Encode conditions
    data["Workload"] = data["Condition"].map({
        "1-back": 0,
        "4-back": 1
    })

    # Define feature, target, and participant groups
    feature_data = data[
        ["Sample", "Workload", "Fz"]
    ].dropna()

    X = feature_data[["Fz"]]
    y = feature_data["Workload"]
    groups = feature_data["Sample"]

    # Leave one sample out cross-validation
    #  tests your model on an unseen sample
    logo = LeaveOneGroupOut()

    y_true = []
    y_pred = []
    y_prob = []

    for train_idx, test_idx in logo.split(X, y, groups):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = LogisticRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # Probability of being 4-back
        probabilities = model.predict_proba(X_test)[:, 1]

        y_true.extend(y_test)
        y_pred.extend(predictions)
        y_prob.extend(probabilities)

        # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # Calculate ROC-AUC
    auc = roc_auc_score(y_true, y_prob)

    # Plot ROC curve
    plt.figure(figsize=(7, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"Fz Relative Theta (AUC = {auc:.2f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Chance (AUC = 0.50)"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: 1-back vs 4-back")
    plt.legend()
    plt.grid(True)
    plt.savefig('n_back_Fc_roc_auc_curve.svg', dpi=600, format='svg', transparent=True)

    # Train the final model on all the samples (no test portion)
    final_model = LogisticRegression()
    final_model.fit(X, y)

    # Save the model 
    joblib.dump(final_model, "unicorn_Fz_logit_model.joblib")

    return auc
