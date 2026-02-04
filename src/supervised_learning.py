import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os

class SupervisedAgent:
    def __init__(self, model_path="data/knn_model.pkl"):
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self.model = None

    def choose_action(self, state):
        # State is (shuttle_y, dist_state, obs_y)
        if self.model:
            # Create a DataFrame with the same column names as training
            df = pd.DataFrame([state], columns=['shuttle_y', 'dist_state', 'obs_y'])
            return self.model.predict(df)[0]
        return 0 # Default fallback

def train_supervised(dataset_path="data/dataset.csv", model_path="data/knn_model.pkl"):
    if not os.path.exists(dataset_path):
        print("Dataset not found.")
        return

    df = pd.read_csv(dataset_path)

    X = df[['shuttle_y', 'dist_state', 'obs_y']]
    y = df['action']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train k-NN
    # k=3 or 5 is standard. Let's try 5.
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"k-NN Model Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Save model
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

    return clf

if __name__ == "__main__":
    train_supervised()
