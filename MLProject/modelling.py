import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import argparse

# Set nama experiment agar tidak default ke ID 0
mlflow.set_experiment("car_purchased_experiment")

# Autolog harus diaktifkan sebelum training
mlflow.sklearn.autolog()

# Argumen dari command line (parameter MLflow Project)
parser = argparse.ArgumentParser()
parser.add_argument("--n_neighbors", type=int, default=3)
args = parser.parse_args()

# Load data
X_train = pd.read_csv("car_preprocessing/X_train.csv")
X_test = pd.read_csv("car_preprocessing/X_test.csv")
y_train = pd.read_csv("car_preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("car_preprocessing/y_test.csv").values.ravel()

# Bungkus training dan evaluasi dalam mlflow.start_run()
with mlflow.start_run():
    model = KNeighborsClassifier(n_neighbors=args.n_neighbors)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc}")
