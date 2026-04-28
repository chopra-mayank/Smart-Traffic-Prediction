import { useState } from "react";
import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Predict from "./pages/Predict";
import MapView from "./pages/MapView";
import Models from "./pages/Models";
import "./index.css";

const NAV = [
  { to: "/", icon: "📊", label: "Dashboard" },
  { to: "/predict", icon: "🔮", label: "Predict" },
  { to: "/map", icon: "🗺️", label: "Traffic Map" },
  { to: "/models", icon: "🤖", label: "Models" },
];

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-brand">
            <h2>Smart Traffic</h2>
            <span>Prediction System</span>
          </div>
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.to === "/"}
              className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
            >
              <span className="nav-icon">{n.icon}</span>
              <span>{n.label}</span>
            </NavLink>
          ))}
        </aside>

        {/* Main Content */}
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/map" element={<MapView />} />
            <Route path="/models" element={<Models />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
