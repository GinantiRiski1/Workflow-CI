# Gunakan base image Miniconda
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy semua isi dari MLProject (karena context: ./MLProject)
COPY . .

# Install dependencies dari conda.yaml
RUN conda env create -f conda.yaml

# Aktifkan environment conda
SHELL ["conda", "run", "-n", "car_purchased", "/bin/bash", "-c"]

# Jalankan training (jika modelling.py perlu dilatih ulang di container) dan serve model
CMD python modelling.py && \
    LATEST_RUN=$(find /app/mlruns/0/ -mindepth 1 -maxdepth 1 -type d | sort -r | head -1) && \
    mlflow models serve -m "$LATEST_RUN/artifacts/model" -p 5000 --host 0.0.0.0

