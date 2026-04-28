import { useEffect, useState } from "react";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend,
} from "recharts";
import { getAllMetrics, getModels } from "../api";

const METRIC_ICONS = { mae: "📉", rmse: "📊", r2: "🎯", mape: "%" };

function ModelCard({ m }) {
  const r2Color = m.r2 > 0.08 ? "var(--success)" : m.r2 > 0 ? "var(--warning)" : "var(--danger)";

  return (
    <div className="card" style={{ position: "relative" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
        <div>
          <p className="card-title">{m.model}</p>
          <span
            style={{
              fontSize: "0.75rem",
              background: "rgba(59,130,246,0.1)",
              color: "var(--accent)",
              padding: "2px 10px",
              borderRadius: 100,
              fontWeight: 600,
            }}
          >
            Trained ✓
          </span>
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>R² Score</div>
          <div style={{ fontSize: "1.8rem", fontWeight: 800, color: r2Color }}>
            {m.r2?.toFixed(3)}
          </div>
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        {[
          ["MAE", m.mae, "Lower = better"],
          ["RMSE", m.rmse, "Lower = better"],
          ["MSE", m.mse, "Lower = better"],
          ["MAPE", m.mape?.toFixed(1) + "%", "Lower = better"],
        ].map(([k, v, hint]) => (
          <div key={k} style={{ background: "rgba(0,0,0,0.2)", borderRadius: 8, padding: "10px 12px" }}>
            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>{k}</div>
            <div style={{ fontSize: "1rem", fontWeight: 700, marginTop: 2 }}>{typeof v === "number" ? v.toFixed(3) : v}</div>
            <div style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}>{hint}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Models() {
  const [metrics, setMetrics] = useState([]);
  const [modelStatus, setModelStatus] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getAllMetrics(), getModels()])
      .then(([m, s]) => {
        setMetrics(m.data.models || []);
        setModelStatus(s.data.models || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  if (loading)
    return (
      <div className="loading-center">
        <div className="spinner" />
        <span>Loading model data…</span>
      </div>
    );

  // Recharts comparison data
  const barData = metrics.map((m) => ({
    name: m.model.replace("_", "\n"),
    MAE: +m.mae?.toFixed(3),
    RMSE: +m.rmse?.toFixed(3),
    R2: +m.r2?.toFixed(3),
  }));

  const radarData = [
    { metric: "Low MAE", ...Object.fromEntries(metrics.map((m) => [m.model, +(1 / (m.mae + 0.01)).toFixed(4)])) },
    { metric: "Low RMSE", ...Object.fromEntries(metrics.map((m) => [m.model, +(1 / (m.rmse + 0.01)).toFixed(4)])) },
    { metric: "High R²", ...Object.fromEntries(metrics.map((m) => [m.model, +Math.max(0, m.r2).toFixed(4)])) },
  ];

  const COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"];

  return (
    <div>
      <div className="page-header">
        <h1>Model Comparison</h1>
        <p>Performance evaluation across all trained regression models</p>
      </div>

      {/* Status badges */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 28 }}>
        {modelStatus.map((s) => (
          <div
            key={s.name}
            style={{
              padding: "8px 16px",
              borderRadius: 100,
              background: s.trained ? "rgba(16,185,129,0.1)" : "rgba(100,116,139,0.1)",
              border: `1px solid ${s.trained ? "rgba(16,185,129,0.3)" : "rgba(100,116,139,0.2)"}`,
              fontSize: "0.8rem",
              fontWeight: 600,
              color: s.trained ? "var(--success)" : "var(--text-muted)",
            }}
          >
            {s.trained ? "✅" : "⏳"} {s.name}
          </div>
        ))}
      </div>

      {/* Model Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 20, marginBottom: 32 }}>
        {metrics.map((m) => <ModelCard key={m.model} m={m} />)}
      </div>

      {/* Bar Chart */}
      <div className="grid-2" style={{ marginBottom: 24 }}>
        <div className="card">
          <p className="card-title">MAE vs RMSE Comparison</p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={barData} barCategoryGap="30%">
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="name" tick={{ fill: "#64748b", fontSize: 10 }} />
              <YAxis tick={{ fill: "#64748b", fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: "#1a2235", border: "1px solid #1e3a5f", borderRadius: 8 }}
                itemStyle={{ color: "#f1f5f9" }}
              />
              <Legend wrapperStyle={{ color: "#94a3b8", fontSize: 12 }} />
              <Bar dataKey="MAE" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              <Bar dataKey="RMSE" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <p className="card-title">R² Score Comparison</p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={barData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="name" tick={{ fill: "#64748b", fontSize: 10 }} />
              <YAxis tick={{ fill: "#64748b", fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: "#1a2235", border: "1px solid #1e3a5f", borderRadius: 8 }}
              />
              <Bar dataKey="R2" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Info box */}
      <div className="card" style={{ background: "rgba(59,130,246,0.06)", borderColor: "var(--border-accent)" }}>
        <p className="card-title">ℹ️ About These Metrics</p>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem", lineHeight: 1.7, marginTop: 8 }}>
          These models predict traffic speed using only time-based features (hour, day, is_weekend, is_peak_hour).
          The moderate R² scores reflect the inherent complexity of METR-LA sensor data — spatial features
          (sensor graph, adjacent roads) would significantly improve performance. The pipeline is designed to
          easily incorporate such features in future iterations.
        </p>
      </div>
    </div>
  );
}
