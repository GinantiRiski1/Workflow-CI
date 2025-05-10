import os
import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import dagshub  # Pastikan dagshub sudah diinstal

# ❗ Tambahkan kredensial autentikasi ke DagsHub
os.environ["MLFLOW_TRACKING_USERNAME"] ="tatiana"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "553123319c3d13c50753c23af6d0f7453f8b1bdb"  # Ganti dengan token asli dari DagsHub

# Inisialisasi DagsHub
dagshub.init(repo_owner='GinantiRiski1', repo_name='my-first-repo', mlflow=True)

# Konfigurasi MLflow untuk menggunakan DagsHub sebagai tracking server
mlflow.set_tracking_uri("https://dagshub.com/GinantiRiski1/my-first-repo.mlflow")

def main(n_neighbors):
    mlflow.sklearn.autolog()

    # Load data
    X_train = pd.read_csv("car_preprocessing/X_train.csv")
    X_test = pd.read_csv("car_preprocessing/X_test.csv")
    y_train = pd.read_csv("car_preprocessing/y_train.csv").values.ravel()
    y_test = pd.read_csv("car_preprocessing/y_test.csv").values.ravel()

    # Set eksperimen
    mlflow.set_experiment("model1")

    # Mulai pencatatan MLflow
    with mlflow.start_run():
        model = KNeighborsClassifier(n_neighbors=int(n_neighbors))
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_neighbors", type=int, default=3, help="Number of neighbors for KNN")
    args = parser.parse_args()

    main(args.n_neighbors)
