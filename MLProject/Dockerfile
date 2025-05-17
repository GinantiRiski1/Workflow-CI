# Gunakan image resmi Miniconda yang stabil
FROM continuumio/miniconda3:latest

# Set working directory di dalam container
WORKDIR /app

# Salin semua file proyek ke dalam image
COPY . .

# Salin environment file dan buat environment conda
COPY conda.yaml .
RUN conda env create -f conda.yaml

# Aktifkan environment conda secara default di shell
SHELL ["conda", "run", "-n", "mlflow-car", "/bin/bash", "-c"]

# Install pip requirements jika ada tambahan (optional)
# RUN pip install -r requirements.txt  # Kalau kamu masih pakai file ini

# Jalankan script training menggunakan conda environment
CMD ["conda", "run", "--no-capture-output", "-n", "mlflow-car", "python", "modelling.py"]

