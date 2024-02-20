from matplotlib import pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.inspection import permutation_importance
import pandas as pd


def calculate_feature_importance(csv_file_healthy, csv_file_swollen, feature_prefix, n_splits=10):
    # Load datasets
    df_swollen = pd.read_csv(csv_file_swollen)
    df_healthy = pd.read_csv(csv_file_healthy)

    # Drop the 'Image' column to retain only numerical values
    df_swollen.drop('Image', axis=1, inplace=True)
    df_healthy.drop('Image', axis=1, inplace=True)

    # Add a label column to each dataset
    df_swollen['label'] = 1  # Class 1
    df_healthy['label'] = 0  # Class 2

    # Combine the datasets
    df_combined = pd.concat([df_swollen, df_healthy])

    # Separate features and labels
    X = df_combined.iloc[:, :-1]  # Features
    y = df_combined['label']  # Labels

    # Create a StratifiedKFold object to maintain the proportion of each class
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Create an SVM classifier
    clf = svm.SVC(kernel='linear', random_state=42)

    # List to store feature importances from each fold
    feature_importances = np.zeros(X.shape[1])

    # Perform k-fold cross-validation and calculate feature importance
    for train_index, test_index in skf.split(X, y):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        clf.fit(X_train_fold, y_train_fold)
        perm_importance = permutation_importance(clf, X_test_fold, y_test_fold, n_repeats=10, random_state=42)
        # Accumulate the feature importances
        feature_importances += perm_importance.importances_mean

    # Average the feature importances over all folds
    feature_importances /= n_splits

    # Sort the features by importance
    sorted_indices = np.argsort(feature_importances)[::-1]

    # Plot the feature importances
    plt.figure(figsize=(12, 8))
    plt.title(f"Permutation Feature Importance with KFold for {feature_prefix}")
    plt.bar(range(len(feature_importances)), feature_importances[sorted_indices], align="center")
    plt.xticks(range(len(feature_importances)), X.columns[sorted_indices], rotation=90)
    plt.ylabel("Mean accuracy decrease across folds")
    plt.tight_layout()# Load datasets
    df_swollen = pd.read_csv(csv_file_swollen)
    df_healthy = pd.read_csv(csv_file_healthy)

    # Drop the 'Image' column to retain only numerical values
    df_swollen.drop('Image', axis=1, inplace=True)
    df_healthy.drop('Image', axis=1, inplace=True)

    # Add a label column to each dataset
    df_swollen['label'] = 1  # Class 1
    df_healthy['label'] = 0  # Class 2

    # Combine the datasets
    df_combined = pd.concat([df_swollen, df_healthy])

    # Separate features and labels
    X = df_combined.iloc[:, :-1]  # Features
    y = df_combined['label']  # Labels

    # Create a StratifiedKFold object to maintain the proportion of each class
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Create an SVM classifier
    clf = svm.SVC(kernel='linear', random_state=42)

    # List to store feature importances from each fold
    feature_importances = np.zeros(X.shape[1])

    # Perform k-fold cross-validation and calculate feature importance
    for train_index, test_index in skf.split(X, y):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        clf.fit(X_train_fold, y_train_fold)
        perm_importance = permutation_importance(clf, X_test_fold, y_test_fold, n_repeats=10, random_state=42)
        # Accumulate the feature importances
        feature_importances += perm_importance.importances_mean

    # Average the feature importances over all folds
    feature_importances /= n_splits

    # Sort the features by importance
    sorted_indices = np.argsort(feature_importances)[::-1]

    # Plot the feature importances
    plt.figure(figsize=(12, 8))
    plt.title(f"Permutation Feature Importance with KFold for {feature_prefix}")
    plt.bar(range(len(feature_importances)), feature_importances[sorted_indices], align="center")
    plt.xticks(range(len(feature_importances)), X.columns[sorted_indices], rotation=90)
    plt.ylabel("Mean accuracy decrease across folds")
    plt.tight_layout()
    plt.savefig(f'results/plots/{feature_prefix}_kfold_feature_importance.png')

    # Print the sorted features and their importance
    print(f"Feature importances with KFold for {feature_prefix}:")
    for idx in sorted_indices:
        print(f"{X.columns[idx]}: {feature_importances[idx]:.4f}")

# Call the function for PIP features
calculate_feature_importance('results/features_pip_healthy.csv', 'results/features_pip_swollen.csv', 'PIP')

# Call the function for DIP features
calculate_feature_importance('results/features_dip_healthy.csv', 'results/features_dip_swollen.csv', 'DIP')
