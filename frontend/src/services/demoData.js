// Central source for all simulated, deterministic frontend demonstration data.
// Nothing in this module contacts a backend or external market-data provider.
export const DEMO_DISCLAIMER = "All displayed market prices, portfolio metrics, forecasts, and optimization outputs are illustrative demonstration values. They are not live market data or investment advice.";

const stockSpecs = [
  { symbol: "RELIANCE", company: "Reliance Industries", sector: "Energy & Retail", price: 1_428.4, change: 0.82, drift: 0.00048, wave: 0.0081, phase: 0.4, color: "#58d6a0" },
  { symbol: "TCS", company: "Tata Consultancy Services", sector: "Information Technology", price: 3_862.75, change: -0.34, drift: 0.00039, wave: 0.0068, phase: 1.7, color: "#85a9ff" },
  { symbol: "INFY", company: "Infosys", sector: "Information Technology", price: 1_542.2, change: 0.56, drift: 0.00043, wave: 0.0095, phase: 2.6, color: "#c39cff" },
  { symbol: "HDFCBANK", company: "HDFC Bank", sector: "Financial Services", price: 1_684.1, change: 0.21, drift: 0.00031, wave: 0.0059, phase: 3.4, color: "#f0bd73" },
  { symbol: "ICICIBANK", company: "ICICI Bank", sector: "Financial Services", price: 1_196.55, change: 1.04, drift: 0.00045, wave: 0.0072, phase: 4.2, color: "#f38c88" },
];

const dateAt = (index) => new Date(Date.UTC(2024, 0, 2 + index)).toISOString().slice(0, 10);
const round = (value, digits = 6) => Number(value.toFixed(digits));
const makeReturns = ({ drift, wave, phase }, count = 180) => Array.from({ length: count }, (_, i) =>
  drift + wave * Math.sin(i * 0.31 + phase) + wave * 0.42 * Math.cos(i * 0.13 + phase * 1.8) + 0.0017 * Math.sin(i * 0.91 + phase),
);

export const DEMO_STOCKS = stockSpecs.map((spec) => {
  const returns = makeReturns(spec);
  let price = spec.price * 0.81;
  const historicalPrices = returns.map((dailyReturn, index) => {
    price *= 1 + dailyReturn;
    return { date: dateAt(index), price: round(price, 2) };
  });
  return { ...spec, change: round(spec.change, 2), returns: returns.map((value) => round(value)), historicalPrices };
});

export const DEMO_PORTFOLIO = {
  investedAmount: 1_000_000,
  allocations: { RELIANCE: 25, TCS: 20, INFY: 20, HDFCBANK: 20, ICICIBANK: 15 },
};

export function getDemoHealth() {
  return { status: "demo", runtime_environment: "static_demo", s3_connectivity: "not_connected", cloud_execution: { verified: false }, message: "Static demonstration data is active." };
}

export function getDemoStocks() {
  return {
    symbols: DEMO_STOCKS.map(({ symbol }) => symbol),
    supported_stock_count: DEMO_STOCKS.length,
    history: { start_date: "2024-01-02", end_date: "2024-06-29", row_count: 180 },
    data_source: { name: "Hardcoded simulated demonstration data", simulated: true, sha256_verified: false },
    stocks: DEMO_STOCKS.map(({ symbol, company, sector, price, change, historicalPrices }) => ({ symbol, company, sector, price, daily_change_percent: change, historical_prices: historicalPrices })),
  };
}

const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
const sampleStd = (values) => {
  const average = mean(values);
  return Math.sqrt(values.reduce((sum, value) => sum + (value - average) ** 2, 0) / Math.max(values.length - 1, 1));
};
const quantile = (values, q) => [...values].sort((a, b) => a - b)[Math.floor((values.length - 1) * q)];
const normalizedWeights = (symbols, weights) => {
  const sum = symbols.reduce((total, symbol) => total + Math.max(0, Number(weights[symbol]) || 0), 0);
  return Object.fromEntries(symbols.map((symbol) => [symbol, sum ? Math.max(0, Number(weights[symbol]) || 0) / sum : 1 / symbols.length]));
};

export function calculateDemoRisk(symbols, weights, startDate = "2024-01-02", endDate = "2024-06-29") {
  const chosen = DEMO_STOCKS.filter((stock) => symbols.includes(stock.symbol));
  if (!chosen.length) return null;
  const indices = chosen[0].returns.map((_, index) => index).filter((index) => dateAt(index) >= startDate && dateAt(index) <= endDate);
  if (indices.length < 2) return null;
  const w = normalizedWeights(chosen.map(({ symbol }) => symbol), weights);
  const returns = indices.map((i) => chosen.reduce((sum, stock) => sum + stock.returns[i] * w[stock.symbol], 0));
  const average = mean(returns);
  const volatility = sampleStd(returns) * Math.sqrt(252);
  const downside = Math.sqrt(mean(returns.map((value) => Math.min(0, value) ** 2))) * Math.sqrt(252);
  let wealth = 1;
  let peak = 1;
  let maxDrawdown = 0;
  const daily_returns = returns.map((value, index) => {
    wealth *= 1 + value;
    peak = Math.max(peak, wealth);
    maxDrawdown = Math.min(maxDrawdown, wealth / peak - 1);
    return { date: dateAt(indices[index]), return: round(value) };
  });
  const var95 = quantile(returns, 0.05);
  const var99 = quantile(returns, 0.01);
  const corr = chosen.map((stockA) => chosen.map((stockB) => {
    const a = stockA.returns;
    const b = stockB.returns;
    const ma = mean(a), mb = mean(b);
    const covariance = mean(a.map((value, i) => (value - ma) * (b[i] - mb)));
    return round(covariance / (sampleStd(a) * sampleStd(b)));
  }));
  const totalReturn = wealth - 1;
  const cagr = wealth ** (252 / returns.length) - 1;
  return {
    period: { start_date: dateAt(indices[0]), end_date: dateAt(indices.at(-1)) },
    metrics: {
      observation_count: returns.length, total_return: round(totalReturn), cagr: round(cagr),
      annualized_volatility: round(volatility), sharpe_ratio: round(average / (sampleStd(returns) || 1) * Math.sqrt(252)),
      sortino_ratio: round(average * 252 / (downside || 1)), beta: round(0.78 + volatility * 0.35),
      maximum_drawdown: round(maxDrawdown), var_95: round(-var95), cvar_95: round(-mean(returns.filter((value) => value <= var95))),
      var_99: round(-var99), cvar_99: round(-mean(returns.filter((value) => value <= var99))),
    },
    daily_returns,
    correlation: { symbols: chosen.map(({ symbol }) => symbol), values: corr },
    allocations: w,
    portfolio: {
      invested_amount: DEMO_PORTFOLIO.investedAmount,
      current_value: round(DEMO_PORTFOLIO.investedAmount * wealth, 2),
      total_return: round(totalReturn),
      asset_allocations: chosen.map(({ symbol }) => ({ symbol, weight: w[symbol] })),
      individual_returns: chosen.map((stock) => ({ symbol: stock.symbol, total_return: round(stock.returns.reduce((value, daily) => value * (1 + daily), 1) - 1) })),
    },
    data_source: { name: "Simulated deterministic demonstration series", simulated: true },
    cloud_execution: { verified: false, data_source: "Hardcoded demo data" },
  };
}

const FORECAST_METHODS = ["Rolling 20-day", "EWMA", "GARCH(1,1)", "Ridge Regression", "Random Forest", "XGBoost"];
const FORECAST_BASE = { "Rolling 20-day": 0.226, EWMA: 0.214, "GARCH(1,1)": 0.231, "Ridge Regression": 0.219, "Random Forest": 0.224, XGBoost: 0.217 };
export const DEMO_MODEL_SCORES = [
  { model: "Rolling 20-day", mae: 0.031, rmse: 0.044, qlike: 0.184 },
  { model: "EWMA", mae: 0.028, rmse: 0.041, qlike: 0.169 },
  { model: "GARCH(1,1)", mae: 0.026, rmse: 0.039, qlike: 0.158 },
  { model: "Ridge Regression", mae: 0.024, rmse: 0.036, qlike: 0.147 },
  { model: "Random Forest", mae: 0.022, rmse: 0.035, qlike: 0.139 },
  { model: "XGBoost", mae: 0.023, rmse: 0.034, qlike: 0.142 },
];
export const DEMO_FORECAST_METHODS = FORECAST_METHODS;

export function getDemoForecast(symbol, method) {
  const stock = DEMO_STOCKS.find((item) => item.symbol === symbol) ?? DEMO_STOCKS[0];
  const base = FORECAST_BASE[method] ?? FORECAST_BASE["Ridge Regression"];
  const shift = (DEMO_STOCKS.findIndex((item) => item.symbol === stock.symbol) - 2) * 0.009;
  return {
    symbol: stock.symbol, model: method, annualized_volatility: round(base + shift, 4),
    chart: Array.from({ length: 20 }, (_, i) => ({ date: `T+${i + 1}`, value: round(base + shift + Math.sin(i * 0.53 + stock.phase) * 0.018, 4) })),
    comparison: DEMO_MODEL_SCORES.map((item) => ({ ...item, mae: round(item.mae + shift * 0.12, 4), rmse: round(item.rmse + shift * 0.16, 4) })),
    evaluation: { basis: "Illustrative precomputed comparison values", target: "20-session realized volatility", trained_in_browser: false },
  };
}

export const DEMO_STRATEGIES = ["Equal Weight", "Minimum Variance", "Maximum Sharpe", "CVaR Optimization", "Forecast-Aware Minimum Variance"];
const strategyWeights = {
  "Equal Weight": [0.2, 0.2, 0.2, 0.2, 0.2],
  "Minimum Variance": [0.14, 0.18, 0.16, 0.31, 0.21],
  "Maximum Sharpe": [0.29, 0.13, 0.24, 0.12, 0.22],
  "CVaR Optimization": [0.16, 0.2, 0.17, 0.29, 0.18],
  "Forecast-Aware Minimum Variance": [0.18, 0.18, 0.2, 0.26, 0.18],
};

export function getDemoOptimization(strategy, symbols = DEMO_STOCKS.map(({ symbol }) => symbol)) {
  const indexBySymbol = Object.fromEntries(DEMO_STOCKS.map(({ symbol }, index) => [symbol, index]));
  const chosen = symbols.filter((symbol) => indexBySymbol[symbol] !== undefined);
  const results = (strategy === "Compare strategies" ? DEMO_STRATEGIES : [strategy]).map((name) => {
    const original = strategyWeights[name] ?? strategyWeights["Equal Weight"];
    const weights = normalizedWeights(chosen, Object.fromEntries(chosen.map((symbol) => [symbol, original[indexBySymbol[symbol]]])));
    const expected = 0.108 + Object.entries(weights).reduce((sum, [symbol, weight]) => sum + weight * (indexBySymbol[symbol] - 2) * 0.006, 0);
    const volatility = 0.176 - Object.values(weights).reduce((sum, weight) => sum + weight ** 2, 0) * 0.018;
    return { strategy: name, weights, expected_return: round(expected, 4), volatility: round(volatility, 4), sharpe: round(expected / volatility, 3), maximum_drawdown: round(-volatility * 0.72, 4), source: "Precomputed illustrative strategy profile" };
  });
  return results;
}
