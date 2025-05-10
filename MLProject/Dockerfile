# Gunakan image base untuk Python
FROM python:3.10-slim

# Set working directory dalam container
WORKDIR /app

# Salin conda.yaml dan requirements ke dalam container
COPY MLProject/conda.yaml /app/conda.yaml

# Install Miniconda (optional, bisa diganti jika sudah menggunakan conda)
RUN apt-get update && apt-get install -y \
    wget \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm Miniconda3-latest-Linux-x86_64.sh \
    && /opt/conda/bin/conda init bash

# Set environment variable untuk menggunakan conda
ENV PATH=/opt/conda/bin:$PATH

# Install dependencies menggunakan conda
RUN conda env create -f /app/conda.yaml

# Set environment untuk menggunakan environment conda
SHELL ["conda", "run", "-n", "mlflow-env", "/bin/bash", "-c"]

# Salin seluruh isi MLProject ke dalam container
COPY MLProject /app

# Install MLflow dan dependencies lainnya
RUN pip install -r /app/MLProject/requirements.txt

# Expose port untuk akses jika diperlukan (misal untuk API atau service)
EXPOSE 5000

# Set entrypoint untuk menjalankan model
ENTRYPOINT ["conda", "run", "-n", "mlflow-env", "python", "/app/MLProject/modelling.py"]
