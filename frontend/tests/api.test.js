import assert from "node:assert/strict";
import test from "node:test";
import { analyzeRisk, getHealth, getResearchExperiments, getStockHistory, optimizePortfolio, runForecast } from "../src/services/api.js";

test("health request uses the configured local API prefix", async () => {
  let observed;
  const result = await getHealth({
    fetchImpl: async (url, options) => {
      observed = { url, options };
      return new Response(JSON.stringify({ status: "healthy" }), { status: 200 });
    },
  });
  assert.equal(observed.url, "/api/health");
  assert.equal(observed.options.method, "GET");
  assert.deepEqual(result, { status: "healthy" });
});

test("risk request posts a validated portfolio to the analysis route", async () => {
  const body = { symbols: ["TCS"], weights: { TCS: 1 } };
  let observed;
  await analyzeRisk(body, {
    fetchImpl: async (url, options) => {
      observed = { url, options };
      return new Response(JSON.stringify({ result_key: "gold/api-results/risk/id.json" }), { status: 200 });
    },
  });
  assert.equal(observed.url, "/api/portfolio/analyze");
  assert.equal(observed.options.method, "POST");
  assert.deepEqual(JSON.parse(observed.options.body), body);
});

test("history and analytics clients target the deployed API routes", async () => {
  const requests = [];
  const fetchImpl = async (url, options) => {
    requests.push({ url, options });
    return new Response("{}", { status: 200 });
  };
  await getStockHistory("RELIANCE", { start_date: "2020-01-01", end_date: "2024-12-31" }, { fetchImpl });
  await runForecast({ symbol: "RELIANCE", model: "ridge" }, { fetchImpl });
  await optimizePortfolio({ symbols: ["RELIANCE"], strategy: "minimum_variance" }, { fetchImpl });
  await getResearchExperiments({ fetchImpl });
  assert.deepEqual(requests.map(({ url }) => url), [
    "/api/stocks/RELIANCE/history?start_date=2020-01-01&end_date=2024-12-31",
    "/api/ml/forecast",
    "/api/portfolio/optimize",
    "/api/research/experiments",
  ]);
  assert.deepEqual(requests.map(({ options }) => options.method), ["GET", "POST", "POST", "GET"]);
});

test("API errors remain explicit to the caller", async () => {
  await assert.rejects(
    getHealth({
      fetchImpl: async () => new Response(JSON.stringify({ error: "S3 unavailable" }), { status: 503 }),
    }),
    /S3 unavailable/,
  );
});
