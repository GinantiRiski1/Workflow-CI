from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from flask import Response

# Metrik Prometheus yang digunakan
REQUEST_COUNT = Counter("request_count", "Total number of requests")
REQUEST_ERRORS = Counter("request_errors", "Total number of errors")
INFERENCE_TIME = Histogram("inference_time_seconds", "Time spent on inference")

# Endpoint metrics
def prometheus_metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
