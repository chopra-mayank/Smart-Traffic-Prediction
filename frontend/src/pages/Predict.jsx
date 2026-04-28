import { useState } from "react";
import { predict } from "../api";

const DAYS = [
  "Monday (0)", "Tuesday (1)", "Wednesday (2)", "Thursday (3)",
  "Friday (4)", "Saturday (5)", "Sunday (6)",
];

const MODELS = ["xgboost", "random_forest", "gradient_boosting", "linear_regression"];

function CongestionBadge({ level }) {
  const cls = {
    "Free Flow": "badge-free",
    Moderate: "badge-moderate",
    Heavy: "badge-heavy",
    Severe: "badge-severe",
  }[level] || "badge-moderate";
  return <span className={`badge ${cls}`} style={{ fontSize: "0.9rem", padding: "6px 16px" }}>● {level}</span>;
}

export default function Predict() {
  const [form, setForm] = useState({
    hour: 8,
    day: 0,
    is_weekend: 0,
    is_peak_hour: 1,
    model_name: "xgboost",
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((f) => {
      const updated = { ...f, [name]: isNaN(value) ? value : Number(value) };
      // auto-derive helpers
      if (name === "hour") {
        updated.is_peak_hour = [7, 8, 9, 17, 18, 19].includes(Number(value)) ? 1 : 0;
      }
      if (name === "day") {
        updated.is_weekend = Number(value) >= 5 ? 1 : 0;
      }
      return updated;
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await predict(form);
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="page-header">
        <h1>Traffic Prediction</h1>
        <p>Enter time-based features to predict traffic speed</p>
      </div>

      <div className="grid-2">
        {/* Form */}
        <div className="card">
          <p className="card-title" style={{ marginBottom: 20 }}>Prediction Inputs</p>
          <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 18 }}>
            <div className="form-group">
              <label className="form-label">Hour of Day (0–23)</label>
              <input
                id="hour"
                name="hour"
                type="number"
                min={0} max={23}
                value={form.hour}
                onChange={handleChange}
                className="form-input"
              />
              <small style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>
                Peak hours: 7–9 AM, 5–7 PM
              </small>
            </div>

            <div className="form-group">
              <label className="form-label">Day of Week</label>
              <select
                id="day"
                name="day"
                value={form.day}
                onChange={handleChange}
                className="form-select"
              >
                {DAYS.map((d, i) => (
                  <option key={i} value={i}>{d}</option>
                ))}
              </select>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              <div className="form-group">
                <label className="form-label">Is Weekend?</label>
                <select
                  id="is_weekend"
                  name="is_weekend"
                  value={form.is_weekend}
                  onChange={handleChange}
                  className="form-select"
                >
                  <option value={0}>No (0)</option>
                  <option value={1}>Yes (1)</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Is Peak Hour?</label>
                <select
                  id="is_peak_hour"
                  name="is_peak_hour"
                  value={form.is_peak_hour}
                  onChange={handleChange}
                  className="form-select"
                >
                  <option value={0}>No (0)</option>
                  <option value={1}>Yes (1)</option>
                </select>
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Model</label>
              <select
                id="model_name"
                name="model_name"
                value={form.model_name}
                onChange={handleChange}
                className="form-select"
              >
                {MODELS.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>

            <button className="btn btn-primary" disabled={loading} id="predict-btn">
              {loading ? "Predicting…" : "⚡ Predict Traffic Speed"}
            </button>
          </form>
        </div>

        {/* Result */}
        <div>
          {error && <div className="error-box" style={{ marginBottom: 16 }}>⚠️ {error}</div>}

          {result ? (
            <div className="pred-result">
              <p style={{ color: "var(--text-muted)", fontSize: "0.8rem", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.1em" }}>
                Predicted Speed
              </p>
              <div className="pred-speed">
                {result.predicted_speed?.toFixed(2)}
                <span className="pred-unit"> mph</span>
              </div>
              <div style={{ marginTop: 16 }}>
                <CongestionBadge level={result.congestion_level} />
              </div>
              <div style={{ marginTop: 24, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, textAlign: "left" }}>
                {[
                  ["Model", result.model_used],
                  ["Hour", `${result.hour}:00`],
                  ["Day", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][result.day]],
                  ["Weekend", result.is_weekend ? "Yes" : "No"],
                  ["Peak Hour", result.is_peak_hour ? "Yes" : "No"],
                ].map(([k, v]) => (
                  <div key={k} style={{ background: "rgba(0,0,0,0.2)", borderRadius: 8, padding: "10px 14px" }}>
                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>{k}</div>
                    <div style={{ fontSize: "0.95rem", fontWeight: 600, marginTop: 4 }}>{v}</div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="card" style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: 300, gap: 12 }}>
              <div style={{ fontSize: "3rem" }}>🔮</div>
              <p style={{ color: "var(--text-muted)" }}>Fill in the form and click Predict</p>
            </div>
          )}

          {/* Congestion Legend */}
          <div className="card" style={{ marginTop: 16 }}>
            <p className="card-title" style={{ marginBottom: 12 }}>Congestion Scale</p>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {[
                ["Free Flow", "≥ 55 mph", "badge-free"],
                ["Moderate", "40–54 mph", "badge-moderate"],
                ["Heavy", "25–39 mph", "badge-heavy"],
                ["Severe", "< 25 mph", "badge-severe"],
              ].map(([label, range, cls]) => (
                <div key={label} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span className={`badge ${cls}`}>● {label}</span>
                  <span style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>{range}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
