
'In machine learning and predictive modeling, unequal error costs (or misclassification costs) arise when different types of errors have different consequences or costs. For example, in a fraud detection system, false positives (i.e., incorrectly flagging a legitimate transaction as fraudulent) may be less costly than false negatives (i.e., failing to detect actual fraud). In such cases, it’s crucial to adjust your model evaluation or decision-making process to account for these different error costs.

#Approach 1: Modify Evaluation Metrics 
'In situations with unequal error costs, a weighted confusion matrix can be useful to assess the impact of each type of misclassification.

import numpy as np
from sklearn.metrics import confusion_matrix

# Define costs for misclassifications
C_FP = 10  # Cost of false positive
C_FN = 50  # Cost of false negative
C_TP = 0   # Cost of true positive
C_TN = 0   # Cost of true negative

# True labels and predictions
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 0])

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Extract the counts from the confusion matrix
TP = cm[1, 1]  # True Positive
TN = cm[0, 0]  # True Negative
FP = cm[0, 1]  # False Positive
FN = cm[1, 0]  # False Negative

# Calculate total misclassification cost
total_cost = (C_FP * FP) + (C_FN * FN) + (C_TP * TP) + (C_TN * TN)
print(f"Total Misclassification Cost: {total_cost}")

#Approach 2: Cost-Sensitive Learning 
'Many machine learning algorithms can be modified to account for unequal misclassification costs by adjusting the class weights or loss function.

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Create a sample imbalanced classification problem
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, 
                            class_sep=2, weights=[0.9, 0.1], random_state=42)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Assign higher weight to the minority class (class 1)
model = RandomForestClassifier(class_weight={0: 1, 1: 10}, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Predictions and confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

#Approach 3: Adjusting Decision Thresholds
'In some cases, you may want to adjust the decision threshold of the classifier to minimize the cost. For example, instead of using the default threshold of 0.5, you can change it depending on the cost associated with false positives and false negatives.

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Find the optimal threshold based on cost
# Suppose False Positive Cost is 5, False Negative Cost is 20
costs = 5 * fpr + 20 * (1 - tpr)
optimal_threshold = thresholds[np.argmin(costs)]

# Predict with the adjusted threshold
y_pred_custom = (y_probs >= optimal_threshold).astype(int)

# Evaluate with confusion matrix
cm = confusion_matrix(y_test, y_pred_custom)
print("Adjusted Confusion Matrix:")
print(cm)

#Approach 4: Weighted Loss Function
'In some machine learning algorithms (like neural networks, logistic regression, or support vector machines), you can directly specify a weighted loss function that penalizes different types of errors.

import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Create an imbalanced dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, 
                            class_sep=2, weights=[0.9, 0.1], random_state=42)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize XGBoost with a weighted loss function
model = xgb.XGBClassifier(scale_pos_weight=10)  # Penalize class 1 more heavily

# Train the model
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

