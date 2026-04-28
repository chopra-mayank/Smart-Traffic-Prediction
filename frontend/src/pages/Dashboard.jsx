import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, AreaChart, Area, Legend,
} from "recharts";
import { predictRealtime, getWeeklyPattern, getAllMetrics } from "../api";

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

function CongestionDot({ level }) {
  const cls = {
    "Free Flow": "badge-free",
    Moderate: "badge-moderate",
    Heavy: "badge-heavy",
    Severe: "badge-severe",
  }[level] || "badge-moderate";
  return <span className={`badge ${cls}`}>● {level}</span>;
}

export default function Dashboard() {
  const [realtime, setRealtime] = useState(null);
  const [weekly, setWeekly] = useState([]);
  const [metrics, setMetrics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      predictRealtime("xgboost"),
      getWeeklyPattern("xgboost"),
      getAllMetrics(),
    ])
      .then(([rt, wp, m]) => {
        setRealtime(rt.data);
        setWeekly(wp.data.weekly_pattern || []);
        setMetrics(m.data.models || []);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });
  }, []);

  if (loading)
    return (
      <div className="loading-center">
        <div className="spinner" />
        <span>Loading dashboard…</span>
      </div>
    );

  if (error)
    return (
      <div className="error-box">
        ⚠️ Could not connect to API: {error}
        <br />
        <small>Make sure the FastAPI server is running at http://localhost:8000</small>
      </div>
    );

  const bestModel = metrics.reduce(
    (best, m) => (!best || m.r2 > best.r2 ? m : best),
    null
  );

  return (
    <div>
      <div className="page-header">
        <h1>Traffic Dashboard</h1>
        <p>Real-time and historical traffic predictions powered by ML</p>
      </div>

      {/* KPIs */}
      <div className="kpi-grid">
        <div className="kpi-card">
          <span className="kpi-icon">🚗</span>
          <span className="kpi-label">Current Speed</span>
          <span className="kpi-value" style={{ color: "var(--accent)" }}>
            {realtime?.predicted_speed?.toFixed(1) ?? "--"}{" "}
            <small style={{ fontSize: "0.9rem", fontWeight: 500 }}>mph</small>
          </span>
          {realtime && <CongestionDot level={realtime.congestion_level} />}
        </div>

        <div className="kpi-card">
          <span className="kpi-icon">🕐</span>
          <span className="kpi-label">Current Hour</span>
          <span className="kpi-value">{realtime?.hour ?? "--"}:00</span>
          <span className="kpi-change">
            {realtime?.is_peak_hour ? "⚠️ Peak Hour" : "✅ Off-Peak"}
          </span>
        </div>

        <div className="kpi-card">
          <span className="kpi-icon">📅</span>
          <span className="kpi-label">Day</span>
          <span className="kpi-value" style={{ fontSize: "1.4rem" }}>
            {realtime ? DAYS[realtime.day] : "--"}
          </span>
          <span className="kpi-change">
            {realtime?.is_weekend ? "Weekend" : "Weekday"}
          </span>
        </div>

        {bestModel && (
          <div className="kpi-card">
            <span className="kpi-icon">🏆</span>
            <span className="kpi-label">Best Model R²</span>
            <span className="kpi-value" style={{ color: "var(--success)" }}>
              {bestModel.r2?.toFixed(3)}
            </span>
            <span className="kpi-change">{bestModel.model}</span>
          </div>
        )}
      </div>

      {/* Weekly Pattern Chart */}
      <div className="grid-2" style={{ marginBottom: 24 }}>
        <div className="card">
          <p className="card-title">Weekly Traffic Pattern</p>
          <ResponsiveContainer width="100%" height={240}>
            <AreaChart data={weekly}>
              <defs>
                <linearGradient id="colorSpeed" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="day" tick={{ fill: "#64748b", fontSize: 11 }} />
              <YAxis tick={{ fill: "#64748b", fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: "#1a2235", border: "1px solid #1e3a5f", borderRadius: 8 }}
                labelStyle={{ color: "#94a3b8" }}
                itemStyle={{ color: "#3b82f6" }}
              />
              <Area
                type="monotone"
                dataKey="avg_speed"
                stroke="#3b82f6"
                fill="url(#colorSpeed)"
                strokeWidth={2}
                name="Avg Speed"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Model Metrics Table */}
        <div className="card">
          <p className="card-title">Model Performance</p>
          <table className="data-table" style={{ marginTop: 8 }}>
            <thead>
              <tr>
                <th>Model</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>R²</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((m) => (
                <tr key={m.model}>
                  <td style={{ color: "var(--text-primary)", fontWeight: 500 }}>
                    {m.model}
                  </td>
                  <td>{m.mae?.toFixed(3)}</td>
                  <td>{m.rmse?.toFixed(3)}</td>
                  <td style={{ color: m.r2 > 0.08 ? "var(--success)" : "var(--danger)" }}>
                    {m.r2?.toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
