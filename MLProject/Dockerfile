FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy all project files to container
COPY . .

# Install dependencies from conda.yaml
RUN conda env create -f conda.yaml

# Activate the environment
SHELL ["conda", "run", "-n", "mlflow-env", "/bin/bash", "-c"]

# Install Flask (pastikan flask ada juga di conda.yaml jika belum)
RUN pip install flask

# Expose the port for Flask
EXPOSE 5000

# Run the Flask app
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mlflow-env", "python", "inference.py"]
