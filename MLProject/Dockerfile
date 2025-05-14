# Gunakan image dasar resmi untuk MLflow
FROM python:3.10-slim

# Install conda (optional, kalau kamu pakai conda env)
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda clean -afy

# Tambahkan conda ke PATH
ENV PATH="/opt/conda/bin:$PATH"

# Salin isi folder project ke dalam container
WORKDIR /app
COPY . /app

# Install dependencies dari conda environment
RUN conda env create -f conda.yaml

# Aktifkan environment MLflow dan jalankan script training
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mlflow-env", "python", "modelling.py"]
