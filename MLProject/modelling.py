import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Argument parser untuk menerima parameter dari MLproject
parser = argparse.ArgumentParser()
parser.add_argument("--n_neighbors", type=int, default=3)
args = parser.parse_args()

# Autolog harus diaktifkan sebelum training
mlflow.sklearn.autolog()

# Load data
X_train = pd.read_csv("car_preprocessing/X_train.csv")
X_test = pd.read_csv("car_preprocessing/X_test.csv")
y_train = pd.read_csv("car_preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("car_preprocessing/y_test.csv").values.ravel()

# Set nama eksperimen
mlflow.set_experiment("basic-model_v2")

# Mulai MLflow run
with mlflow.start_run():
    model = KNeighborsClassifier(n_neighbors=args.n_neighbors)
    model.fit(X_train, y_train)

    # Prediksi dan akurasi
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc}")

    # Log manual (jika autolog tidak digunakan)
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_metric("accuracy", acc)
