from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST
from flask import Response

# 1. Total request count per endpoint & method
REQUEST_COUNT = Counter(
    "request_count", "Total number of requests", ['endpoint', 'method']
)

# 2. Total error count per endpoint
REQUEST_ERRORS = Counter(
    "request_errors", "Total number of errors", ['endpoint']
)

# 3. Histogram: time spent on inference
INFERENCE_TIME = Histogram(
    "inference_time_seconds", "Time spent on inference"
)

# 4. Gauge: number of in-flight requests (being processed)
IN_FLIGHT_REQUESTS = Gauge(
    "in_flight_requests", "Number of requests currently being processed"
)

# 5. Summary: request latency by endpoint
LATENCY_SUMMARY = Summary(
    "request_latency_seconds", "Summary of request latencies", ['endpoint']
)

# 6. Gauge: CPU usage percentage (dummy value, to be updated in inference.py)
CPU_USAGE_PERCENT = Gauge(
    "cpu_usage_percent", "Current CPU usage percent"
)

# 7. Gauge: Memory usage percentage (dummy value, to be updated in inference.py)
MEMORY_USAGE_PERCENT = Gauge(
    "memory_usage_percent", "Current Memory usage percent"
)

# 8. Counter: requests grouped by user-agent
REQUESTS_BY_USER_AGENT = Counter(
    "requests_by_user_agent", "Number of requests grouped by User-Agent", ['user_agent']
)

# 9. Histogram: request payload size in bytes
REQUEST_PAYLOAD_SIZE = Histogram(
    "request_payload_size_bytes", "Size of request payload in bytes"
)

# 10. Histogram: response payload size in bytes
RESPONSE_PAYLOAD_SIZE = Histogram(
    "response_payload_size_bytes", "Size of response payload in bytes"
)

def prometheus_metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
