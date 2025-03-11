

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# loading data
df = pd.read_csv("synthetic_transactions.csv")

# Creating features for further model training
X = df[['amount', 'timestamp', 'gas_fee', 'transaction_count', 'wallet_age']]
y = df['is_fraud']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Isolation Forest model
if_model = IsolationForest(contamination=0.1, random_state=42)
if_model.fit(X_train)

# Predict on test set using Isolation Forest
if_pred = if_model.predict(X_test)
if_pred = [1 if pred == -1 else 0 for pred in if_pred]

# Train One-Class SVM model
ocsvm_model = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.1)
ocsvm_model.fit(X_train)

# Predict on test set using One-Class SVM
ocsvm_pred = ocsvm_model.predict(X_test)
ocsvm_pred = [1 if pred == -1 else 0 for pred in ocsvm_pred]

# Evaluate models
if_precision = precision_score(y_test, if_pred)
if_recall = recall_score(y_test, if_pred)
if_f1 = f1_score(y_test, if_pred)

ocsvm_precision = precision_score(y_test, ocsvm_pred)
ocsvm_recall = recall_score(y_test, ocsvm_pred)
ocsvm_f1 = f1_score(y_test, ocsvm_pred)

print(f"Isolation Forest - Precision: {if_precision}, Recall: {if_recall}, F1-score: {if_f1}")
print(f"One-Class SVM - Precision: {ocsvm_precision}, Recall: {ocsvm_recall}, F1-score: {ocsvm_f1}")
