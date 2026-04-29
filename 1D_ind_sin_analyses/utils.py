import os
import h5py
import pywt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.metrics import accuracy_score

def process_file(file_path):
    with h5py.File(file_path, 'r') as File:
        events_means = [[] for _ in range(42)]
        for electrode in list(File.keys()):
            events = list(File[electrode].keys())
            for event in events:
                data = File[electrode][f'{event}']['data'][:]
                events_means[int(event[6:])-1].append(data)
        final_means = [np.mean(event, axis=0) for event in events_means]
    return final_means


def label_lab(a,groups):
    for key in groups.keys():
        if a in groups[key]:
            return key


def experiment_based_classification(df,age):
    """
    Perform LOOCV classification with SVC on datasets built from wavelet_props.
    
    Parameters
    ----------
    wavelet_props : dict-like
        Dictionary or DataFrame-like object containing wavelet properties.
        Must support indexing by integer keys (0..20) and have a 'type' column
        with an .apply(age) method.
    
    Returns
    -------
    results : dict
        Mapping dataset index -> LOOCV accuracy.
    """
    
    # Build datasets
    datasets: List[Tuple[str, np.ndarray, np.ndarray]] = [
        (
            f"{i}",
            np.asarray(df[i]).reshape(-1, 1),
            np.asarray(df['type'].apply(age))
        )
        for i in range(21)
    ]

    # Define pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=False))
    ])

    # Define parameter grid
    param_grid = [
        {"svc__kernel": ["linear"], "svc__C": [0.01, 0.1, 1, 10, 100]},
        {"svc__kernel": ["rbf"], "svc__C": [0.01, 0.1, 1, 10, 100],
         "svc__gamma": ["scale", "auto", 1e-3, 1e-2, 1e-1, 1, 10]},
        {"svc__kernel": ["poly"], "svc__C": [0.01, 0.1, 1, 10, 100],
         "svc__degree": [2, 3, 4, 5],
         "svc__gamma": ["scale", "auto", 1e-3, 1e-2, 1e-1, 1]},
        {"svc__kernel": ["sigmoid"], "svc__C": [0.01, 0.1, 1, 10, 100],
         "svc__gamma": ["scale", "auto", 1e-3, 1e-2, 1e-1, 1]},
    ]

    results = {}

    for n_ex, X, y in datasets:
        loo = LeaveOneOut()
        y_true, y_pred = [], []

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Grid search with inner CV
            grid = GridSearchCV(
                pipeline,
                param_grid,
                cv=3,
                scoring='accuracy',
                n_jobs=-1
            )
            grid.fit(X_train, y_train)

            # Best pipeline from grid search
            best_clf = grid.best_estimator_

            # Predict
            prediction = best_clf.predict(X_test)

            y_true.append(y_test[0])
            y_pred.append(prediction[0])

        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"LOOCV Accuracy {n_ex}: {accuracy:.2%}")
        results[n_ex] = accuracy

    return results