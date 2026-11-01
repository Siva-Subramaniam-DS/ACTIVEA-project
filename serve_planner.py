"""Serve the planner and score calories with the saved model.

The browser cannot load a scikit-learn file directly. This program loads
Model/best_calorie_model.joblib (trained on endomondoHR.csv) and answers
the page at /api/calories.
Open http://127.0.0.1:8767/ after starting it.
"""

from __future__ import annotations

import sys
import json
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

# Scikit-Learn cross-version unpickling compatibility
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
try:
    import sklearn.compose._column_transformer as column_transformer
    if not hasattr(column_transformer, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        column_transformer._RemainderColsList = _RemainderColsList
except Exception:
    pass

try:
    from sklearn._loss import _loss
    sys.modules["_loss"] = _loss
except Exception:
    pass

import joblib

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
MODEL_PATH = ROOT / "Model" / "best_calorie_model.joblib"
MODEL = joblib.load(MODEL_PATH)

# Activity mapping from Coursework Exercises to Endomondo telemetry features
SPORT_MAP = {
    "Exercise 1": ("run", 8.5),
    "Exercise 2": ("core stability training", 2.0),
    "Exercise 3": ("bike", 22.0),
    "Exercise 4": ("core stability training", 1.0),
    "Exercise 5": ("core stability training", 1.0),
    "Exercise 6": ("indoor cycling", 12.0),
    "Exercise 7": ("run", 10.5),
    "Exercise 8": ("walk", 5.5),
    "Exercise 9": ("core stability training", 2.5),
    "Exercise 10": ("orienteering", 9.0),
    "Jogging": ("run", 8.5),
    "Bodyweight squats": ("core stability training", 2.0),
    "Cycling (moderate)": ("bike", 22.0),
    "Plank": ("core stability training", 1.0),
    "Yoga": ("core stability training", 1.0),
    "Rope skipping": ("indoor cycling", 12.0),
    "Running": ("run", 10.5),
    "Brisk walking": ("walk", 5.5),
    "Pushups & Calisthenics": ("core stability training", 2.5),
    "Agility drills & Burpees": ("orienteering", 9.0),
}


def predict_calories(profile: dict, blocks: list[dict]) -> list[float]:
    if not blocks:
        return []
    age = float(profile.get("age", 30.0))
    gender = str(profile.get("gender", "male")).lower()
    resting = float(profile.get("resting", 60.0))

    rows = []
    for block in blocks:
        hr_pct = float(block.get("hr", 70.0))
        duration = float(block.get("duration", 15.0))
        heart_rate = resting + (hr_pct / 100.0) * (220.0 - age - resting)
        ex_key = str(block.get("exercise", "Exercise 1"))

        sport, speed = SPORT_MAP.get(ex_key, ("run", 8.0))
        dist_km = speed * (duration / 60.0)

        rows.append({
            "duration_min": duration,
            "heart_rate_avg": heart_rate,
            "heart_rate_max": heart_rate + 15.0,
            "heart_rate_std": 8.5,
            "speed_avg_kmh": speed,
            "elevation_gain_m": 15.0,
            "distance_km": dist_km,
            "sport": sport,
            "gender": gender
        })

    df_input = pd.DataFrame(rows)
    predictions = MODEL.predict(df_input)
    return [round(float(value), 1) for value in predictions]


class PlannerHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", ""):
            path = "/index.html"
        file_path = (WEB / path.lstrip("/")).resolve()
        if not str(file_path).startswith(str(WEB.resolve())) or not file_path.is_file():
            self.send_error(404)
            return
        kind = "text/html; charset=utf-8"
        if file_path.suffix == ".jpg":
            kind = "image/jpeg"
        elif file_path.suffix == ".json":
            kind = "application/json"
        elif file_path.suffix == ".css":
            kind = "text/css; charset=utf-8"
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if urlparse(self.path).path != "/api/calories":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        try:
            calories = predict_calories(payload.get("profile", {}), payload.get("blocks", []))
        except (KeyError, TypeError, ValueError) as error:
            body = json.dumps({"error": str(error)}).encode("utf-8")
            self.send_response(400)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = json.dumps({"calories": calories})
        encoded = body.encode("utf-8")
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")

    def log_message(self, format, *args):
        return


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 8767), PlannerHandler)
    print("=" * 60)
    print("ACTIVEA Interactive Session Planner Server")
    print(f"Loaded Champion Model: {MODEL_PATH.name}")
    print("Open http://127.0.0.1:8767/ in your browser.")
    print("Press Ctrl+C to stop.")
    print("=" * 60, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
