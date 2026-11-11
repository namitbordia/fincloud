import assert from "node:assert/strict";
import test from "node:test";
import { calculateDemoRisk, DEMO_DISCLAIMER, DEMO_PORTFOLIO, DEMO_STOCKS, DEMO_STRATEGIES, getDemoForecast, getDemoHealth, getDemoOptimization, getDemoStocks } from "../src/services/demoData.js";

test("central demo provider exposes the five-stock sample and illustrative data notice", () => {
  assert.deepEqual(DEMO_STOCKS.map(({ symbol }) => symbol), ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]);
  assert.match(DEMO_DISCLAIMER, /not live market data/);
  assert.equal(getDemoStocks().supported_stock_count, 5);
  assert.equal(getDemoHealth().cloud_execution.verified, false);
});

test("default portfolio allocations match the requested demonstration portfolio", () => {
  assert.deepEqual(DEMO_PORTFOLIO.allocations, { RELIANCE: 25, TCS: 20, INFY: 20, HDFCBANK: 20, ICICIBANK: 15 });
  assert.equal(Object.values(DEMO_PORTFOLIO.allocations).reduce((sum, value) => sum + value, 0), 100);
});

test("risk metrics are deterministic and respond to changed portfolio weights", () => {
  const first = calculateDemoRisk(["RELIANCE", "TCS", "INFY"], { RELIANCE: 50, TCS: 30, INFY: 20 });
  const repeated = calculateDemoRisk(["RELIANCE", "TCS", "INFY"], { RELIANCE: 50, TCS: 30, INFY: 20 });
  const changed = calculateDemoRisk(["RELIANCE", "TCS", "INFY"], { RELIANCE: 10, TCS: 20, INFY: 70 });
  assert.deepEqual(first, repeated);
  assert.notEqual(first.metrics.annualized_volatility, changed.metrics.annualized_volatility);
  assert.equal(first.daily_returns.length, 180);
  assert.equal(first.correlation.values.length, 3);
});

test("forecast results include all six illustrative methods and comparison scores", () => {
  const forecast = getDemoForecast("INFY", "Ridge Regression");
  assert.equal(forecast.chart.length, 20);
  assert.equal(forecast.comparison.length, 6);
  assert.equal(forecast.evaluation.trained_in_browser, false);
  assert.ok(forecast.annualized_volatility > 0);
});

test("each optimization profile is deterministic and normalizes selected allocations", () => {
  assert.equal(DEMO_STRATEGIES.length, 5);
  for (const strategy of DEMO_STRATEGIES) {
    const [result] = getDemoOptimization(strategy, ["RELIANCE", "TCS", "INFY"]);
    assert.equal(result.strategy, strategy);
    assert.ok(Math.abs(Object.values(result.weights).reduce((sum, value) => sum + value, 0) - 1) < 1e-10);
  }
});
