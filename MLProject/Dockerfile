# Gunakan base image Miniconda
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy semua isi dari MLProject (karena context: ./MLProject)
COPY . .

# Install dependencies dari conda.yaml
RUN conda env create -f conda.yaml

# Pakai shell conda run untuk aktifkan environment
SHELL ["conda", "run", "-n", "car_purchased", "/bin/bash", "-c"]

# Copy script start.sh yang berisi perintah untuk training + serve
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Jalankan start.sh sebagai CMD
CMD ["conda", "run", "--no-capture-output", "-n", "car_purchased", "bash", "/app/start.sh"]

