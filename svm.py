from sklearn import svm
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load datasets
df_swollen = pd.read_csv('results/features_pip_swollen.csv')
df_healthy = pd.read_csv('results/features_pip_healthy.csv')

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

# Standardize the features (important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create an SVM classifier
clf = svm.SVC(kernel='linear')  # You can change the kernel and hyperparameters as needed

# Create StratifiedKFold object with 10 splits
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Perform k-fold cross-validation
cv_scores = cross_val_score(clf, X, y, cv=kfold, scoring='accuracy')

# Print the accuracy for each fold
print(f"Accuracies for each fold are: {cv_scores}")

# Print the mean and standard deviation of the cross-validation scores
print(f"Mean cross-validation accuracy: {np.mean(cv_scores):.2f}")
print(f"Standard Deviation of cross-validation accuracy: {np.std(cv_scores):.2f}")

# If you need to save the model for later use
# from joblib import dump
# dump(clf, 'svm_classifier.joblib')
