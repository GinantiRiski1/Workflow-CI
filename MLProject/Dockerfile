FROM continuumio/miniconda3

WORKDIR /app

COPY MLProject /app

RUN conda env create -f conda.yaml

SHELL ["conda", "run", "-n", "msml-env", "/bin/bash", "-c"]

CMD ["mlflow", "run", "."]
