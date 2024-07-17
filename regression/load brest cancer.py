from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

# Create AdaBoost with Logistic Regression as the base estimator
log_reg_boost = AdaBoostClassifier(estimator=LogisticRegression(max_iter=10000),
                                   n_estimators=50, random_state=42)

# Train the model
log_reg_boost.fit(X_train, y_train)

# Make predictions
log_reg_preds = log_reg_boost.predict(X_test)

# Evaluate the model
accuracy_lr = accuracy_score(y_test, log_reg_preds)
precision_lr = precision_score(y_test, log_reg_preds)
recall_lr = recall_score(y_test, log_reg_preds)
f1_lr = f1_score(y_test, log_reg_preds)

print("Logistic Regression Boosting:")
print(f"Accuracy: {accuracy_lr}, Precision: {precision_lr}, Recall: {recall_lr}, F1 Score: {f1_lr}")

from sklearn.tree import DecisionTreeClassifier

# Create AdaBoost with Decision Tree as the base estimator
dt_boost = AdaBoostClassifier(estimator=DecisionTreeClassifier(),
                              n_estimators=50, random_state=42)

# Train the model
dt_boost.fit(X_train, y_train)

# Make predictions
dt_preds = dt_boost.predict(X_test)

# Evaluate the model
accuracy_dt = accuracy_score(y_test, dt_preds)
precision_dt = precision_score(y_test, dt_preds)
recall_dt = recall_score(y_test, dt_preds)
f1_dt = f1_score(y_test, dt_preds)

print("\nDecision Tree Boosting:")
print(f"Accuracy: {accuracy_dt}, Precision: {precision_dt}, Recall: {recall_dt}, F1 Score: {f1_dt}")

