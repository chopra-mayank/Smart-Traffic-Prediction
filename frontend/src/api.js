import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const api = axios.create({ baseURL: BASE_URL, timeout: 10000 });

export const predict = (data) => api.post("/predict", data);
export const predictRealtime = (model = "xgboost") =>
  api.get("/predict/realtime", { params: { model_name: model } });
export const predictBatch = (records) => api.post("/predict/batch", { records });
export const getModels = () => api.get("/models");
export const getAllMetrics = () => api.get("/metrics");
export const getMetrics = (model) => api.get(`/metrics/${model}`);
export const getHeatmap = (model = "xgboost") =>
  api.get("/traffic/heatmap", { params: { model_name: model } });
export const getWeeklyPattern = (model = "xgboost") =>
  api.get("/traffic/weekly-pattern", { params: { model_name: model } });
