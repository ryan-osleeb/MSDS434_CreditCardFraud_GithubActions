#!/usr/bin/env python3
import json, time, pandas as pd
from kafka import KafkaProducer

# Absolute path to fraud-only dataset
CSV = "/home/ubuntu/ccfraud/creditcard_fraud.csv"

# Configure producer (point to your Kafka broker host/port)
producer = KafkaProducer(
    bootstrap_servers="ec2-13-221-27-204.compute-1.amazonaws.com:9092",
    value_serializer=lambda v: json.dumps(v).encode()
)

# Load fraud-only dataset
df = pd.read_csv(CSV)

# Build payloads without "Class" column
features = [c for c in df.columns if c != "Class"]
for i, row in df.iterrows():
    payload = {"id": int(i), "features": {k: float(row[k]) for k in features}}
    producer.send("transactions", payload)
    if i % 1000 == 0:  # flush every 1000 rows
        producer.flush()
        time.sleep(0.01)

producer.flush()
print("done producing.")
