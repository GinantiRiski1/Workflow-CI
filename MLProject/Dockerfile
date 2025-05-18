FROM continuumio/miniconda3:latest

WORKDIR /app

COPY . .

RUN conda env create -f conda.yaml

SHELL ["conda", "run", "-n", "car_purchased", "/bin/bash", "-c"]

# Run training sekaligus serve model dalam satu command shell script
CMD python modelling.py && \
    LATEST_RUN=$(ls -td /app/mlruns/0/* | head -1) && \
    mlflow models serve -m "$LATEST_RUN/artifacts/model" -p 5000 --host 0.0.0.0
