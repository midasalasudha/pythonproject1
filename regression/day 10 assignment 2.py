import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, delimiter=';')

# Binarize the target variable
data['quality'] = (data['quality'] >= 7).astype(int)

# Separate features and target
X = data.drop(columns='quality')
y = data['quality']

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train and evaluate k-NN model
def evaluate_knn(k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Experiment with different values of k
k_values = [3, 5, 7]
accuracies = {k: evaluate_knn(k) for k in k_values}

# Report the accuracy for each value of k
for k, accuracy in accuracies.items():
    print(f'k = {k}, Accuracy = {accuracy:.4f}')