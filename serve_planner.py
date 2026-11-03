"""Serve the planner page and the calorie calculation.

Calories follow the coursework table. The printed rate is for 70 kg, and the
typed body mass scales it: kcal/min = table rate * (body mass / 70).
The saved Endomondo model is not used here. It has no body-mass input, and
mapping yoga or rope skipping onto another sport produced the wrong calories.
Open http://127.0.0.1:8767/ after starting it.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from activea.data_loader import ExerciseKnowledgeBase

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
ACTIVITIES = ExerciseKnowledgeBase.get_default_activities()


def predict_calories(profile: dict, blocks: list[dict]) -> list[float]:
    """Return one calorie total per block, scaled by the typed body mass."""
    if not blocks:
        return []
    weight = float(profile.get("weight", 70.0))
    if weight <= 0:
        raise ValueError("body mass must be positive")
    calories = []
    for block in blocks:
        name = str(block.get("exercise", ""))
        activity = ACTIVITIES.get(name)
        if activity is None:
            raise KeyError(name)
        duration = float(block.get("duration", 0.0))
        if duration < 0:
            raise ValueError("duration must be zero or positive")
        calories.append(round(activity.calories_per_min(weight) * duration, 4))
    return calories


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
        body = json.dumps({"calories": calories, "source": "table"})
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
    print("Calories: coursework table scaled by body mass / 70.")
    print("Open http://127.0.0.1:8767/ in your browser.")
    print("Press Ctrl+C to stop.")
    print("=" * 60, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
