import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import dagshub  # Pastikan dagshub sudah terinstal

# Inisialisasi DagsHub untuk MLflow
dagshub.init(repo_owner='GinantiRiski1', repo_name='my-first-repo', mlflow=True)  # Ganti dengan username dan repo kamu

# Konfigurasi MLflow ke DagsHub
mlflow.set_tracking_uri("https://dagshub.com/GinantiRiski1/my-first-repo.mlflow")  # Ganti dengan URL repo kamu

# Fungsi untuk training model
def main(n_neighbors):
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
        # Model dengan parameter n_neighbors yang diterima
        model = KNeighborsClassifier(n_neighbors=int(n_neighbors))
        model.fit(X_train, y_train)

        # Prediksi dan akurasi
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc}")

if __name__ == "__main__":
    # Parse argumen dari command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_neighbors", type=int, default=3, help="Number of neighbors for KNN")
    args = parser.parse_args()

    # Panggil main function dengan argumen n_neighbors
    main(args.n_neighbors)
