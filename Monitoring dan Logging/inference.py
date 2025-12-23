import time
import pandas as pd
import mlflow.pyfunc
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

EXPERIMENT_ID = "413842369526119047"
RUN_ID = "340d0a52a0b94f0bb918f185a7cc6a01"
model = mlflow.pyfunc.load_model(
    f"./Membangun_model/mlartifacts/{EXPERIMENT_ID}/{RUN_ID}/artifacts/model"
)

REQ_TOTAL = Counter("inference_requests_total", "Total inference requests")
LATENCY = Histogram("inference_latency_seconds", "Inference latency")
PRED_TOTAL = Counter("predictions_total", "Total predictions", ["label"])

TRAIN_COLS = pd.read_csv(
    "./preprocessing/heart_disease_preprocessing/heart_disease_clean.csv"
).drop(columns=["target"]).columns.tolist()

@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()
    payload = request.get_json()

    df = pd.DataFrame(payload)
    df = df.reindex(columns=TRAIN_COLS, fill_value=0)

    preds = model.predict(df)

    elapsed = time.time() - start

    REQ_TOTAL.inc()
    LATENCY.observe(elapsed)
    for p in preds:
        PRED_TOTAL.labels(str(int(p))).inc()

    return jsonify({"predictions": preds.tolist()})

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)



