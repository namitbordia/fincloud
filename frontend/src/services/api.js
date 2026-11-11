const API_BASE = (import.meta.env?.VITE_API_BASE_URL || "/api").replace(/\/$/, "");

export async function apiRequest(path, { method = "GET", body, token, fetchImpl = fetch } = {}) {
  const headers = { Accept: "application/json" };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (token) headers.Authorization = `Bearer ${token}`;
  const response = await fetchImpl(`${API_BASE}${path}`, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await response.text();
  let payload = {};
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      throw new Error(`API returned invalid JSON (HTTP ${response.status}).`);
    }
  }
  if (!response.ok) {
    const message = typeof payload.error === "string" ? payload.error : `API request failed (HTTP ${response.status}).`;
    throw new Error(message);
  }
  return payload;
}

export const getHealth = (options) => apiRequest("/health", options);
export const getStocks = (options) => apiRequest("/stocks", options);
export const getStockHistory = (symbol, query = {}, options) => {
  const params = new URLSearchParams(Object.entries(query).filter(([, value]) => value));
  const suffix = params.size ? `?${params}` : "";
  return apiRequest(`/stocks/${encodeURIComponent(symbol)}/history${suffix}`, options);
};
export const analyzeRisk = (body, options = {}) =>
  apiRequest("/portfolio/analyze", { ...options, method: "POST", body });
export const getCorrelation = (body, options = {}) =>
  apiRequest("/portfolio/correlation", { ...options, method: "POST", body });
export const runForecast = (body, options = {}) =>
  apiRequest("/ml/forecast", { ...options, method: "POST", body });
export const optimizePortfolio = (body, options = {}) =>
  apiRequest("/portfolio/optimize", { ...options, method: "POST", body });
export const getResearchExperiments = (options) => apiRequest("/research/experiments", options);
export const getResearchModels = (options) => apiRequest("/research/models", options);
