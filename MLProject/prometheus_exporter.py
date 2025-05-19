from prometheus_client import Counter, Histogram, generate_latest
from flask import Response

REQUEST_COUNT = Counter("request_count", "Total number of requests")
INFERENCE_TIME = Histogram("inference_time_seconds", "Time spent on prediction")

@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype="text/plain")
