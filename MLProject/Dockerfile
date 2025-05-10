FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy project files
COPY . /app/

# Create conda environment
RUN conda env create -f conda.yaml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "mlflow-env", "/bin/bash", "-c"]

# Set environment variables
ENV MLFLOW_TRACKING_URI: "https://dagshub.com/GinantiRiski1/my-first-repo.mlflow"

# Command to run when container starts
CMD ["conda", "run", "-n", "mlflow-env", "python", "modelling.py"] 
