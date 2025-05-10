import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Aktifkan autolog dari MLflow sebelum training
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
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # Prediksi dan hitung akurasi
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc}")

    # (Opsional) Logging model secara eksplisit ke MLflow
    mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_test[:5])
