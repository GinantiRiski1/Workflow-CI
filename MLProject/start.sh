#!/bin/bash
set -e  # Stop script jika error

echo "Mulai training..."
python modelling.py

echo "Cari model run terbaru..."
LATEST_RUN=$(find /app/mlruns/0/ -mindepth 1 -maxdepth 1 -type d | sort -r | head -1)

if [ -z "$LATEST_RUN" ]; then
  echo "Error: Tidak ditemukan folder run mlflow di /app/mlruns/0/"
  exit 1
fi

echo "Menjalankan model dari $LATEST_RUN/artifacts/model"
mlflow models serve -m "$LATEST_RUN/artifacts/model" -p 5000 --host 0.0.0.0
