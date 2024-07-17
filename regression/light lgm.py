# LightGBM classifier
import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = load_breast_cancer()
X, y = data.data, data.target
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create the LightGBM dataset
dtrain = lgb.Dataset(X_train, label=y_train)
dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)
# Set up the parameters with adjusted num_leaves and other parameters
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 30,  # Increase the number of leaves
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 20,
    'min_sum_hessian_in_leaf': 0.001
}

# Train the LightGBM model
num_round = 20  # Increase the number of boosting rounds
bst = lgb.train(params, dtrain, num_round, valid_sets=[dtrain, dtest])
# Predict on the test set
preds = bst.predict(X_test, num_iteration=bst.best_iteration)
preds = np.round(preds)  # Convert probabilities to binary predictions
# Evaluate the model
accuracy = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy}")
# Save the model
bst.save_model('lightgbm_model.txt')
# Load the model (example of loading)
loaded_bst = lgb.Booster(model_file='lightgbm_model.txt')
# Predict using the loaded model
loaded_preds = loaded_bst.predict(X_test, num_iteration=loaded_bst.best_iteration)
loaded_preds = np.round(loaded_preds)
loaded_accuracy = accuracy_score(y_test, loaded_preds)
print(f"Loaded model accuracy: {loaded_accuracy}")

'''
Contents of 'lightgbm_model.txt'
Model Parameters:
The file will contain the parameters used to train the LightGBM model. These parameters
include various settings.
Feature Names:
If provided during the creation of lgb.Dataset, the file may include the names of the 
features used in training (dtrain.feature_name). This helps in understanding which
features correspond to which columns in the input data.
Tree Structures:
For tree-based models (such as Gradient Boosting Decision Trees, GBDT), the file will 
include the structure of the trees that were trained during the boosting rounds. 
Each tree's structure typically includes nodes, splits, thresholds, and leaf values. 
This information is crucial for making predictions and understanding how the model makes
decisions.
Model Metadata:
Additional metadata about the model, such as the number of boosting rounds (num_round), 
the best iteration based on early stopping (bst.best_iteration), and other relevant 
information that might be useful when loading and using the model.
Format:
The file format is typically text-based and readable. It's designed to store all necessary 
information required to reconstruct and use the trained model later. LightGBM uses a custom
format optimized for its models, ensuring efficient storage and retrieval of model 
information.
'''

