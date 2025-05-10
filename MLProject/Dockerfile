# Menggunakan image dasar Miniconda
FROM continuumio/miniconda3:latest

# Menyeting direktori kerja dalam container
WORKDIR /app

# Menyalin file yang diperlukan (termasuk MLProject dan conda.yaml)
COPY . /app

# Install dependencies dari conda.yaml
RUN conda env create -f MLProject/conda.yaml && \
    conda clean --all -f -y

# Set environment variable untuk conda environment
ENV PATH /opt/conda/envs/mlflow-env/bin:$PATH

# Mengaktifkan environment yang telah dibuat
RUN echo "source activate mlflow-env" > ~/.bashrc
ENV CONDA_DEFAULT_ENV mlflow-env

# Menginstall MLflow dan dependencies lainnya
RUN pip install mlflow==2.22.0

# Menentukan entry point untuk container (menjalankan eksperimen MLflow)
ENTRYPOINT ["mlflow", "run", "."]

# Menentukan port yang digunakan oleh MLflow
EXPOSE 5000
