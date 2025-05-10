# Gunakan image dasar Python
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY MLProject/conda.yaml /app/conda.yaml
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh \
    && /opt/conda/bin/conda init bash

# Update Conda
RUN /opt/conda/bin/conda update -n base -c defaults conda

# Set environment variable
ENV PATH=/opt/conda/bin:$PATH

# Install Conda environment and dependencies from conda.yaml
RUN /opt/conda/bin/conda env create -f /app/conda.yaml

# Activate environment
RUN echo "source activate msml-env" > ~/.bashrc
ENV PATH /opt/conda/envs/msml-env/bin:$PATH

# Install MLflow
RUN pip install mlflow

# Salin project ke dalam container
COPY MLProject /app/MLProject

# Install any other dependencies, if needed
# RUN pip install -r requirements.txt

# Command to run the model after container startup
ENTRYPOINT ["mlflow", "run", "MLProject", "-P", "n_neighbors=3"]
