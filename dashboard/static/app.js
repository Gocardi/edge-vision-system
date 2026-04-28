const statusEl = document.getElementById("stream-status");
const eventsTotalEl = document.getElementById("events-total");
const alertsTotalEl = document.getElementById("alerts-total");
const criticalTotalEl = document.getElementById("critical-total");
const highTotalEl = document.getElementById("high-total");
const cameraIdEl = document.getElementById("camera-id");
const lastEventEl = document.getElementById("last-event");
const feedEl = document.getElementById("event-feed");
const cameraFrameEl = document.getElementById("camera-frame");
const frameTimeEl = document.getElementById("frame-time");

const MAX_FEED_ITEMS = 80;

// Traducciones para mostrar en la UI (español)
const EVENT_TYPE_MAP = {
  no_helmet: "Sin casco",
  no_vest: "Sin chaleco",
  no_helmet_no_vest: "Sin casco y sin chaleco",
  ppe_compliant: "EPP OK",
  clear: "Sin personas"
};

const SEVERITY_MAP = {
  critical: "Crítico",
  high: "Alto",
  none: "OK"
};

function formatDate(isoValue) {
  if (!isoValue) return "-";
  const date = new Date(isoValue);
  if (Number.isNaN(date.getTime())) return isoValue;
  return date.toLocaleString();
}

function updateStats(stats) {
  eventsTotalEl.textContent = stats.events_total ?? 0;
  alertsTotalEl.textContent = stats.alerts_total ?? 0;
  criticalTotalEl.textContent = stats.critical_total ?? 0;
  highTotalEl.textContent = stats.high_total ?? 0;
  cameraIdEl.textContent = stats.camera_id || "-";
  lastEventEl.textContent = formatDate(stats.last_event_at);
}

function createFeedItem(event) {
  const item = document.createElement("article");
  item.className = "feed-item";

  const severity = (event.severity || "none").toLowerCase();
  const typeLabel = EVENT_TYPE_MAP[event.event_type] || event.event_type || "desconocido";
  const badgeText = SEVERITY_MAP[severity] || severity;

  item.innerHTML = `
    <div class="feed-top">
      <strong>${typeLabel}</strong>
      <span class="badge ${severity}">${badgeText}</span>
    </div>
    <div class="topic">${event.topic || "-"}</div>
    <div class="meta">Cámara: ${event.camera_id || "-"} | Fotograma: ${event.frame ?? "-"} | Confianza: ${event.confidence ?? 0}</div>
    <div class="meta">${formatDate(event.timestamp)}</div>
  `;

  return item;
}

function prependEvent(event) {
  const node = createFeedItem(event);
  feedEl.prepend(node);

  while (feedEl.children.length > MAX_FEED_ITEMS) {
    feedEl.removeChild(feedEl.lastChild);
  }
}

function updateFrame(framePayload) {
  if (!framePayload || !framePayload.image_b64) {
    return;
  }
  cameraFrameEl.src = `data:image/jpeg;base64,${framePayload.image_b64}`;
  frameTimeEl.textContent = `Fotograma ${framePayload.frame ?? "-"} | ${formatDate(
    framePayload.timestamp
  )}`;
}

async function loadSnapshot() {
  const response = await fetch("/api/snapshot", { cache: "no-store" });
  const data = await response.json();

  updateStats(data.stats || {});
  feedEl.innerHTML = "";

  if (data.latest_frame) {
    updateFrame(data.latest_frame);
  }

  (data.events || []).forEach((event) => {
    feedEl.appendChild(createFeedItem(event));
  });
}

function setOnline(online) {
  statusEl.textContent = online ? "MQTT stream activo" : "Sin conexion";
  statusEl.classList.toggle("online", online);
  statusEl.classList.toggle("offline", !online);
}

function connectStream() {
  const stream = new EventSource("/api/stream");

  stream.onopen = () => setOnline(true);

  stream.onmessage = (event) => {
    const payload = JSON.parse(event.data);

    if (payload.kind === "frame") {
      updateFrame(payload);
      return;
    }

    prependEvent(payload);

    const current = {
      events_total: Number(eventsTotalEl.textContent || "0") + 1,
      alerts_total:
        Number(alertsTotalEl.textContent || "0") +
        (payload.topic === "edge/alerts" ? 1 : 0),
      critical_total:
        Number(criticalTotalEl.textContent || "0") +
        (payload.severity === "critical" ? 1 : 0),
      high_total:
        Number(highTotalEl.textContent || "0") +
        (payload.severity === "high" ? 1 : 0),
      camera_id: payload.camera_id || "-",
      last_event_at: payload.timestamp || null,
    };

    updateStats(current);
  };

  stream.onerror = () => {
    setOnline(false);
    stream.close();
    setTimeout(connectStream, 2000);
  };
}

loadSnapshot()
  .then(() => connectStream())
  .catch(() => {
    setOnline(false);
    setTimeout(connectStream, 2000);
  });
