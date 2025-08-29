#!/usr/bin/env python3
import json
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from joblib import load
from xgboost import XGBClassifier
from prometheus_client import Counter, Histogram, start_http_server
import time

# ---- Prometheus Metrics ----
messages_consumed = Counter("fraud_messages_total", "Total messages consumed from Kafka")
frauds_detected = Counter("fraud_predictions_total", "Total fraud predictions")
latency = Histogram("fraud_prediction_latency_seconds", "Time spent predicting")

# Start Prometheus exporter on port 8001
start_http_server(8001)

# ---- Load artifacts ----
PREPROC_PATH = "/home/ubuntu/ccfraud/models/preprocessor.joblib"
MODEL_PATH = "/home/ubuntu/ccfraud/models/xgb_model.json"

pre = load(PREPROC_PATH)
clf = XGBClassifier()
clf.load_model(MODEL_PATH)

# ---- Kafka Consumer ----
consumer = KafkaConsumer(
    "transactions",
    bootstrap_servers="ec2-13-221-27-204.compute-1.amazonaws.com:9092",
    value_deserializer=lambda b: json.loads(b.decode()),
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="scorers"
)

producer = KafkaProducer(
    bootstrap_servers="ec2-13-221-27-204.compute-1.amazonaws.com:9092",
    value_serializer=lambda v: json.dumps(v).encode()
)

# ---- Processing Loop ----
for msg in consumer:
    start_time = time.time()
    data = msg.value  # {"id": ..., "features": {...}}

    X = pd.DataFrame([data["features"]])  # one-row DataFrame
    p = float(clf.predict_proba(pre.transform(X))[:, 1][0])

    out = {"id": data["id"], "fraud_probability": p}
    producer.send("scored-transactions", out)

    # Update Prometheus metrics
    messages_consumed.inc()
    if p >= 0.5:  # fraud threshold
        frauds_detected.inc()
    latency.observe(time.time() - start_time)

    print(out)
