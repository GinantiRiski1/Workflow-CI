import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Fungsi untuk training model
def main(alpha):
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
        # Model dengan parameter alpha yang diterima
        model = KNeighborsClassifier(n_neighbors=int(alpha))
        model.fit(X_train, y_train)

        # Prediksi dan akurasi
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc}")

if __name__ == "__main__":
    # Parse argumen dari command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=3, help="Alpha (n_neighbors) for KNN")
    args = parser.parse_args()

    # Panggil main function dengan argumen alpha
    main(args.alpha)

