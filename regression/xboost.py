# XGBoost classifier

import xgboost as xgb
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the dataset into DMatrix, the data structure used by XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set up the parameters
params = {
    'max_depth': 3,                     # Maximum depth of the tree
    'eta': 0.1,                         # Learning rate
    'objective': 'binary:logistic',     # Binary classification
    'eval_metric': 'logloss'            # Evaluation metric
}

# Train the XGBoost model
num_round = 10
bst = xgb.train(params, dtrain, num_round)

# Predict on the test set
preds = bst.predict(dtest)
preds = np.round(preds)  # Convert probabilities to binary predictions

# Evaluate the model
accuracy = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy}")

# Save the model
bst.save_model('xgboost_model.json')

# Load the model (example of loading)
loaded_bst = xgb.Booster()
loaded_bst.load_model('xgboost_model.json')

# Predict using the loaded model
loaded_preds = loaded_bst.predict(dtest)
loaded_preds = np.round(loaded_preds)
loaded_accuracy = accuracy_score(y_test, loaded_preds)
print(f"Loaded model accuracy: {loaded_accuracy}")








