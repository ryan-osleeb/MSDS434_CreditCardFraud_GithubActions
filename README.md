# MSDS434: Cloud Computing Final Project credit card fraud demo (xgboost + kafka streaming + prometheous/grafana)

train and serve a credit-card fraud detector using xgboost on an aws ec2 instance.  
the system provides both a rest api (fastapi) and a kafka streaming path with prometheus metrics.

data from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## demo video

[▶ watch the demo](https://github.com/ryan-osleeb/MSDS434_FinalProject_CreditCardFraud/releases/latest/download/MSDS434_CreditCardFraud_Application_default.mp4
)
---

## requirements

- aws ec2 instance (tested on c5.large, ubuntu 22.04)
- python 3.10+ with venv
- apache kafka 3.9.1
- java 17 runtime (for kafka)
- aws cli
- grafana + prometheus (for monitoring)

### system packages (ubuntu)
```bash
sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip unzip wget default-jre
java -version   # confirm java 17+

# download and unpack kafka
curl -O https://downloads.apache.org/kafka/3.9.1/kafka_2.13-3.9.1.tgz
tar -xzf kafka_2.13-3.9.1.tgz

# start zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# start broker
bin/kafka-server-start.sh config/server.properties

# test a topic
bin/kafka-console-producer.sh --topic quickstart-events --bootstrap-server localhost:9092
bin/kafka-console-consumer.sh --topic quickstart-events --from-beginning --bootstrap-server localhost:9092
```

### start FastAPI
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8080
```

## install prometheus & grafana (python)
```bash
pip install prometheus-client grafanalib
```


