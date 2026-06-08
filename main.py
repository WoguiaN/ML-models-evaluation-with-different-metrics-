import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (confusion_matrix )
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px

np.random.seed(42)

# Generate random data
X = np.random.rand(200, 4)  # Input features
y = np.random.randint(0, 2, size=200)  # Binary target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create logistic regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model's performance
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# True Positives
tp = cm[1, 1]
# False Positives
fp = cm[0, 1]
# False Negatives
fn = cm[1, 0]
# True Negatives
tn = cm[0, 0]

# Accuracy
accuracy = (tp + tn) / (tp + tn + fp + fn)
# Precision
precision = tp / (tp + fp)
# Recall
recall = tp / (tp + fn)
# Specificity
specificity = tn / (tn + fp)
# F1 Score
f1 = 2 * (precision * recall) / (precision + recall)


# Print the evaluation metrics
print("Confusion Matrix:", cm)
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print("Specificity:", specificity)
print('F1 Score:', f1)