FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY templates ./templates
COPY static ./static

ENV MQTT_BROKER=mqtt
ENV MQTT_PORT=1883
ENV EVENTS_TOPIC=camera/events
ENV ALERTS_TOPIC=edge/alerts
ENV ACTIONS_TOPIC=edge/actions
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=8080

CMD ["python", "app.py"]
