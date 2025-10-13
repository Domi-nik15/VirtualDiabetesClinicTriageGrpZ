# serve/serve.py
import os
import joblib
from flask import Flask, request, jsonify

MODEL_NAME  = os.getenv("MODEL_NAME", "diabetes-progression")
MODEL_STAGE = os.getenv("MODEL_STAGE", "staging")
MODEL_VERSION = os.getenv("MODEL_VERSION", "0.0")

model = joblib.load(f"artifacts/{MODEL_NAME}/model.pkl")

app = Flask(__name__)

@app.get("/health")
def health():
    return {"status": "ok", "model": f"{MODEL_NAME}", "model_version": f"{MODEL_VERSION}"}

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Expected a JSON object"), 400

    rec = [
        float(payload.get("age",0)),
        float(payload.get("sex",0)),
        float(payload.get("bmi", 0)),
        float(payload.get("bp", 0)),
        float(payload.get("s1", 0)),
        float(payload.get("s2", 0)),
        float(payload.get("s3", 0)),
        float(payload.get("s4", 0)),
        float(payload.get("s5", 0)),
        float(payload.get("s6", 0)),
    ]

    try:
        y = model.predict([rec])
        return jsonify({"prediction": float(y[0])})
    except Exception as e:
        return jsonify(error=type(e).__name__, detail=str(e), rec=rec), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696)
