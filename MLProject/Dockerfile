# Gunakan base image miniconda yang sudah terinstall Conda
FROM continuumio/miniconda3

# Salin seluruh folder MLProject ke dalam container di direktori /MLProject
COPY MLProject/ /MLProject

# Set folder kerja (working directory) di dalam container ke /MLProject
WORKDIR /MLProject

# Perbarui conda ke versi terbaru dan install dependencies berdasarkan conda.yaml
RUN conda update -n base -c defaults conda -y
RUN conda env create -f conda.yaml

# Aktifkan environment msml-env
RUN echo "source activate msml-env" > ~/.bashrc

# Install mlflow di environment msml-env
RUN conda activate msml-env && pip install mlflow

# Set perintah untuk menjalankan MLflow secara otomatis
ENTRYPOINT ["conda", "run", "-n", "msml-env", "mlflow", "run", ".", "-P", "n_neighbors=3"]
