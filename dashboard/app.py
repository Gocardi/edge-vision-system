import json
import os
import queue
import threading
import time
from collections import deque
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
from flask import Flask, Response, jsonify, render_template

app = Flask(__name__)

MQTT_BROKER = os.getenv("MQTT_BROKER", "mqtt")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
EVENTS_TOPIC = os.getenv("EVENTS_TOPIC", "camera/events")
ALERTS_TOPIC = os.getenv("ALERTS_TOPIC", "edge/alerts")
ACTIONS_TOPIC = os.getenv("ACTIONS_TOPIC", "edge/actions")
FRAMES_TOPIC = os.getenv("FRAMES_TOPIC", "camera/frames")

MAX_EVENTS = 150
metrics_lock = threading.Lock()
recent_events = deque(maxlen=MAX_EVENTS)

stats = {
    "events_total": 0,
    "alerts_total": 0,
    "critical_total": 0,
    "high_total": 0,
    "camera_id": "-",
    "last_event_at": None,
}

subscribers_lock = threading.Lock()
subscribers = []
latest_frame_lock = threading.Lock()
latest_frame = None


def parse_iso_timestamp(ts):
    if not ts or not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def serialize_event(topic, payload):
    event = {
        "topic": topic,
        "camera_id": payload.get("camera_id", "-"),
        "event_type": payload.get("event_type", "unknown"),
        "severity": payload.get("severity", "none"),
        "confidence": payload.get("confidence", 0),
        "timestamp": payload.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "source": payload.get("source", "unknown"),
    }

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    event["frame"] = metadata.get("frame", "-")
    event["helmet"] = metadata.get("helmet", {})
    event["vest"] = metadata.get("vest", {})
    return event


def publish_to_subscribers(data):
    with subscribers_lock:
        dead_queues = []
        for q in subscribers:
            try:
                q.put_nowait(data)
            except queue.Full:
                dead_queues.append(q)

        for q in dead_queues:
            if q in subscribers:
                subscribers.remove(q)


def update_stats(event):
    with metrics_lock:
        stats["events_total"] += 1
        stats["camera_id"] = event.get("camera_id", "-")
        stats["last_event_at"] = event.get("timestamp")

        if event.get("topic") == ALERTS_TOPIC:
            stats["alerts_total"] += 1

        severity = event.get("severity", "none")
        if severity == "critical":
            stats["critical_total"] += 1
        elif severity == "high":
            stats["high_total"] += 1

        recent_events.appendleft(event)


def on_connect(client, userdata, flags, rc):
    if rc != 0:
        return
    client.subscribe(EVENTS_TOPIC, qos=1)
    client.subscribe(ALERTS_TOPIC, qos=1)
    client.subscribe(ACTIONS_TOPIC, qos=1)
    client.subscribe(FRAMES_TOPIC, qos=0)


def on_message(client, userdata, message):
    try:
        payload = json.loads(message.payload.decode("utf-8"))
    except json.JSONDecodeError:
        return

    if message.topic == FRAMES_TOPIC:
        frame_payload = {
            "kind": "frame",
            "camera_id": payload.get("camera_id", "-"),
            "timestamp": payload.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "frame": payload.get("frame", "-"),
            "image_b64": payload.get("image_b64", ""),
        }
        with latest_frame_lock:
            global latest_frame
            latest_frame = frame_payload
        publish_to_subscribers(frame_payload)
        return

    event = serialize_event(message.topic, payload)
    update_stats(event)
    publish_to_subscribers(event)


def mqtt_worker():
    client = mqtt.Client(client_id="edge-dashboard")
    client.on_connect = on_connect
    client.on_message = on_message

    while True:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            client.loop_forever()
        except Exception:
            time.sleep(3)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/snapshot")
def api_snapshot():
    with metrics_lock:
        payload = {
            "stats": dict(stats),
            "events": list(recent_events),
        }
    with latest_frame_lock:
        payload["latest_frame"] = dict(latest_frame) if latest_frame else None
    return jsonify(payload)


@app.route("/api/stream")
def api_stream():
    q = queue.Queue(maxsize=20)

    with subscribers_lock:
        subscribers.append(q)

    def event_stream():
        try:
            while True:
                try:
                    data = q.get(timeout=15)
                    yield f"data: {json.dumps(data)}\n\n"
                except queue.Empty:
                    yield "event: ping\ndata: {}\n\n"
        finally:
            with subscribers_lock:
                if q in subscribers:
                    subscribers.remove(q)

    return Response(event_stream(), mimetype="text/event-stream")


if __name__ == "__main__":
    mqtt_thread = threading.Thread(target=mqtt_worker, daemon=True)
    mqtt_thread.start()

    app.run(
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", "8080")),
        debug=False,
    )
