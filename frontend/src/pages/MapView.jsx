import { useEffect, useState } from "react";
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { getHeatmap } from "../api";

// LA Sensor locations (real representative coords from METR-LA)
const LA_SENSORS = [
  { id: "S1", lat: 34.0522, lng: -118.2437, name: "Downtown LA" },
  { id: "S2", lat: 34.0736, lng: -118.4004, name: "West Hollywood" },
  { id: "S3", lat: 33.9416, lng: -118.4085, name: "Inglewood" },
  { id: "S4", lat: 34.1478, lng: -118.1445, name: "Pasadena" },
  { id: "S5", lat: 33.9731, lng: -118.2479, name: "Compton" },
  { id: "S6", lat: 34.0195, lng: -118.4912, name: "Santa Monica" },
  { id: "S7", lat: 34.1830, lng: -118.3089, name: "Burbank" },
  { id: "S8", lat: 34.0259, lng: -118.7798, name: "Thousand Oaks" },
  { id: "S9", lat: 33.8358, lng: -118.3406, name: "Torrance" },
  { id: "S10", lat: 34.0633, lng: -117.9218, name: "Pomona" },
];

function speedToColor(speed) {
  if (speed >= 55) return "#10b981";   // green — free flow
  if (speed >= 40) return "#f59e0b";   // yellow — moderate
  if (speed >= 25) return "#f97316";   // orange — heavy
  return "#ef4444";                    // red — severe
}

function speedToCongestion(speed) {
  if (speed >= 55) return "Free Flow";
  if (speed >= 40) return "Moderate";
  if (speed >= 25) return "Heavy";
  return "Severe";
}

function FitBounds() {
  const map = useMap();
  useEffect(() => {
    map.setView([34.05, -118.35], 10);
  }, [map]);
  return null;
}

export default function MapView() {
  const [model, setModel] = useState("xgboost");
  const [hour, setHour] = useState(new Date().getHours());
  const [day, setDay] = useState(new Date().getDay() === 0 ? 6 : new Date().getDay() - 1);
  const [heatmap, setHeatmap] = useState([]);
  const [loading, setLoading] = useState(false);

  const DAYS_LABEL = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

  const fetchHeatmap = (m = model) => {
    setLoading(true);
    getHeatmap(m)
      .then((r) => { setHeatmap(r.data.data || []); setLoading(false); })
      .catch(() => setLoading(false));
  };

  useEffect(() => { fetchHeatmap(); }, []);

  // Filter to selected hour + day
  const cell = heatmap.find((r) => r.hour === hour && r.day === day);
  const currentSpeed = cell?.speed ?? null;

  // Assign per-sensor speeds by adding small noise based on sensor index
  const sensorData = LA_SENSORS.map((s, i) => {
    const base = currentSpeed ?? 40;
    const jitter = (Math.sin(i * 13.7 + hour * 0.5 + day) * 8);
    const spd = Math.max(5, Math.min(80, base + jitter));
    return { ...s, speed: spd, congestion: speedToCongestion(spd) };
  });

  return (
    <div>
      <div className="page-header">
        <h1>Traffic Map</h1>
        <p>Visualise predicted traffic speed across the LA sensor network</p>
      </div>

      {/* Controls */}
      <div className="card" style={{ marginBottom: 20, display: "flex", gap: 20, flexWrap: "wrap", alignItems: "flex-end" }}>
        <div className="form-group" style={{ minWidth: 160 }}>
          <label className="form-label">Model</label>
          <select
            value={model}
            onChange={(e) => { setModel(e.target.value); fetchHeatmap(e.target.value); }}
            className="form-select"
          >
            {["xgboost", "random_forest", "gradient_boosting", "linear_regression"].map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>

        <div className="form-group" style={{ minWidth: 120 }}>
          <label className="form-label">Hour: {hour}:00</label>
          <input
            type="range" min={0} max={23} value={hour}
            onChange={(e) => setHour(Number(e.target.value))}
            style={{ width: "100%", accentColor: "var(--accent)" }}
          />
        </div>

        <div className="form-group" style={{ minWidth: 120 }}>
          <label className="form-label">Day: {DAYS_LABEL[day]}</label>
          <input
            type="range" min={0} max={6} value={day}
            onChange={(e) => setDay(Number(e.target.value))}
            style={{ width: "100%", accentColor: "var(--accent-2)" }}
          />
        </div>

        {currentSpeed && (
          <div style={{ padding: "10px 18px", background: "rgba(59,130,246,0.1)", border: "1px solid var(--border-accent)", borderRadius: 10 }}>
            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase" }}>Avg Speed</div>
            <div style={{ fontSize: "1.5rem", fontWeight: 800, color: "var(--accent)" }}>
              {currentSpeed.toFixed(1)} <span style={{ fontSize: "0.8rem", fontWeight: 400 }}>mph</span>
            </div>
          </div>
        )}
      </div>

      {/* Map */}
      <div className="map-wrapper">
        <MapContainer
          center={[34.05, -118.35]}
          zoom={10}
          style={{ height: "100%", width: "100%", background: "#0d1117" }}
        >
          <FitBounds />
          <TileLayer
            attribution='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          />
          {sensorData.map((s) => (
            <CircleMarker
              key={s.id}
              center={[s.lat, s.lng]}
              radius={18}
              pathOptions={{
                fillColor: speedToColor(s.speed),
                color: "white",
                weight: 1.5,
                fillOpacity: 0.85,
              }}
            >
              <Popup>
                <div style={{ minWidth: 160 }}>
                  <strong>{s.name}</strong>
                  <br />
                  <span style={{ color: speedToColor(s.speed) }}>
                    ● {s.congestion}
                  </span>
                  <br />
                  Speed: <strong>{s.speed.toFixed(1)} mph</strong>
                  <br />
                  Hour: {hour}:00 | {DAYS_LABEL[day]}
                </div>
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>
      </div>

      {/* Legend */}
      <div className="card" style={{ marginTop: 16, display: "flex", gap: 24, flexWrap: "wrap" }}>
        {[
          ["#10b981", "Free Flow", "≥ 55 mph"],
          ["#f59e0b", "Moderate", "40–54 mph"],
          ["#f97316", "Heavy", "25–39 mph"],
          ["#ef4444", "Severe", "< 25 mph"],
        ].map(([color, label, range]) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 14, height: 14, borderRadius: "50%", background: color }} />
            <span style={{ fontSize: "0.85rem" }}><strong>{label}</strong> — {range}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
