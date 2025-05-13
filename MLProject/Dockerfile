FROM continuumio/miniconda3

WORKDIR /app

COPY . .

RUN conda env create -f conda.yaml

SHELL ["conda", "run", "-n", "mlflow-env", "/bin/bash", "-c"]

# Update ENTRYPOINT untuk menjalankan inference.py
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mlflow-env", "python", "inference.py"]
