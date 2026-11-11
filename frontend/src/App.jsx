import { useEffect, useMemo, useState } from "react";
import DashboardPage from "./pages/DashboardPage.jsx";
import PortfolioBuilderPage from "./pages/PortfolioBuilderPage.jsx";
import RiskAnalyticsPage from "./pages/RiskAnalyticsPage.jsx";
import ForecastingPage from "./pages/ForecastingPage.jsx";
import OptimizationPage from "./pages/OptimizationPage.jsx";
import CloudArchitecturePage from "./pages/CloudArchitecturePage.jsx";
import ResearchPage from "./pages/ResearchPage.jsx";
import { calculateDemoRisk, DEMO_DISCLAIMER, DEMO_PORTFOLIO, getDemoForecast, getDemoHealth, getDemoOptimization, getDemoStocks } from "./services/demoData.js";
import { analyzeRisk, getHealth, getResearchExperiments, getResearchModels, getStockHistory, getStocks, optimizePortfolio, runForecast } from "./services/api.js";
import { About, Architecture, Forecast, Market, Optimization, Overview, Portfolio, Research, Risk } from "./pages/LivePages.jsx";

const NAV_ITEMS = [
  ["dashboard", "Overview", "O"], ["market", "Market overview", "M"], ["builder", "Portfolio analysis", "P"],
  ["risk", "Risk analytics", "R"], ["forecast", "ML forecasting", "F"], ["optimization", "Optimization", "O"],
  ["research", "Research lab", "L"], ["architecture", "Cloud architecture", "C"], ["about", "About FinCloud", "A"],
];
const DEMO_NAV = NAV_ITEMS.filter(([id]) => !["market", "about"].includes(id));
const DEMO_STOCKS = getDemoStocks();
const INITIAL_SYMBOLS = Object.keys(DEMO_PORTFOLIO.allocations);

export default function App() {
  const [mode, setMode] = useState("connecting");
  const [page, setPage] = useState("dashboard");
  const [apiProblem, setApiProblem] = useState("");
  const [busy, setBusy] = useState(false);
  const [apiHealth, setApiHealth] = useState(null);
  const [stocks, setStocks] = useState(null);
  const [selected, setSelected] = useState(INITIAL_SYMBOLS);
  const [weights, setWeights] = useState(DEMO_PORTFOLIO.allocations);
  const [startDate, setStartDate] = useState("2024-01-02");
  const [endDate, setEndDate] = useState("2024-06-29");
  const [riskResult, setRiskResult] = useState(() => calculateDemoRisk(INITIAL_SYMBOLS, DEMO_PORTFOLIO.allocations));
  const [forecastResult, setForecastResult] = useState(() => getDemoForecast("RELIANCE", "Ridge Regression"));
  const [optimizationResult, setOptimizationResult] = useState(() => getDemoOptimization("Minimum Variance", INITIAL_SYMBOLS));
  const [forecastSymbol, setForecastSymbol] = useState("RELIANCE");
  const [forecastModel, setForecastModel] = useState("ridge");
  const [strategy, setStrategy] = useState("minimum_variance");
  const [optimizationSymbols, setOptimizationSymbols] = useState(INITIAL_SYMBOLS);
  const [experiments, setExperiments] = useState(null);
  const [models, setModels] = useState(null);
  const [marketSymbol, setMarketSymbol] = useState("RELIANCE");
  const [history, setHistory] = useState(null);
  const demoHealth = useMemo(getDemoHealth, []);

  async function connect() {
    setMode("connecting");
    setApiProblem("");
    try {
      const [health, universe] = await Promise.all([getHealth(), getStocks()]);
      if (health.status !== "healthy" || !health.cloud_execution?.verified) throw new Error("AWS Lambda or private S3 data verification failed.");
      setApiHealth(health);
      setStocks(universe);
      setSelected(universe.symbols);
      setWeights(Object.fromEntries(universe.symbols.map((symbol) => [symbol, 100 / universe.symbols.length])));
      setOptimizationSymbols(universe.symbols);
      setForecastSymbol(universe.symbols[0]);
      setMarketSymbol(universe.symbols[0]);
      setStartDate(universe.history.start_date);
      setEndDate(universe.history.end_date);
      setMode("live");
    } catch (error) {
      setApiProblem(error.message || "The FinCloud API is unavailable.");
      setMode("error");
    }
  }

  useEffect(() => { connect(); }, []);

  async function apiAction(action) {
    setBusy(true);
    setApiProblem("");
    try { await action(); } catch (error) { setApiProblem(error.message || "API request failed."); }
    finally { setBusy(false); }
  }

  function navigate(next) {
    setPage(next);
    setApiProblem("");
    if (mode !== "live") return;
    if (next === "research") apiAction(async () => {
      const [summary, registry] = await Promise.all([getResearchExperiments(), getResearchModels()]);
      setExperiments(summary); setModels(registry);
    });
    if (next === "market") loadHistory();
  }

  function loadHistory() {
    if (!stocks) return;
    apiAction(async () => setHistory(await getStockHistory(marketSymbol, { start_date: startDate, end_date: endDate })));
  }

  function runRisk() {
    apiAction(async () => {
      const result = await analyzeRisk({
        symbols: selected,
        weights: Object.fromEntries(selected.map((symbol) => [symbol, Number(weights[symbol]) / 100])),
        start_date: startDate,
        end_date: endDate,
      });
      setRiskResult(result); setPage("risk");
    });
  }

  function runLiveForecast() {
    apiAction(async () => setForecastResult(await runForecast({ symbol: forecastSymbol, model: forecastModel })));
  }

  function runLiveOptimization() {
    apiAction(async () => setOptimizationResult(await optimizePortfolio({
      symbols: optimizationSymbols, strategy, start_date: startDate, end_date: endDate,
    })));
  }

  function toggleDemoPortfolioSymbol(symbol) {
    const next = selected.includes(symbol) ? selected.filter((item) => item !== symbol) : [...selected, symbol];
    setSelected(next);
    setWeights((current) => {
      const preserved = Object.fromEntries(next.map((item) => [item, current[item] ?? 0]));
      if (next.length && !Object.values(preserved).some((value) => Number(value) > 0)) return Object.fromEntries(next.map((item) => [item, 100 / next.length]));
      return preserved;
    });
  }

  function demoPageContent() {
    switch (page) {
      case "builder": return <PortfolioBuilderPage symbols={DEMO_STOCKS.symbols} details={DEMO_STOCKS.stocks} investedAmount={DEMO_PORTFOLIO.investedAmount} selected={selected} weights={weights} startDate={startDate} endDate={endDate} setStartDate={setStartDate} setEndDate={setEndDate} onToggle={toggleDemoPortfolioSymbol} onWeight={(symbol, value) => setWeights((current) => ({ ...current, [symbol]: value }))} onAnalyze={() => { const result = calculateDemoRisk(selected, Object.fromEntries(selected.map((symbol) => [symbol, Number(weights[symbol]) || 0])), startDate, endDate); if (result) { setRiskResult(result); setPage("risk"); } }} />;
      case "risk": return <RiskAnalyticsPage result={riskResult} onBuild={() => setPage("builder")} />;
      case "forecast": return <ForecastingPage symbols={DEMO_STOCKS.symbols} result={forecastResult} model="Ridge Regression" setModel={() => {}} symbol="RELIANCE" setSymbol={() => {}} onRun={() => {}} />;
      case "optimization": return <OptimizationPage symbols={DEMO_STOCKS.symbols} selected={optimizationSymbols} onToggle={(symbol) => setOptimizationSymbols((current) => current.includes(symbol) ? current.filter((item) => item !== symbol) : [...current, symbol])} strategy="Minimum Variance" setStrategy={() => {}} onRun={() => {}} result={optimizationResult} />;
      case "research": return <ResearchPage />;
      case "architecture": return <CloudArchitecturePage health={demoHealth} stocks={DEMO_STOCKS} />;
      default: return <DashboardPage stocks={DEMO_STOCKS} onBuild={() => setPage("builder")} onNavigate={setPage} />;
    }
  }

  function livePageContent() {
    if (!stocks) return null;
    const symbols = stocks.symbols;
    const stockDetails = { ...stocks, stocks: Object.fromEntries(symbols.map((symbol) => [symbol, { ...stocks.stocks[symbol], company: symbol, sector: "NSE equity" }])) };
    switch (page) {
      case "market": return <Market stocks={stockDetails} symbol={marketSymbol} setSymbol={setMarketSymbol} startDate={startDate} setStartDate={setStartDate} endDate={endDate} setEndDate={setEndDate} onLoad={loadHistory} history={history} busy={busy} />;
      case "builder": return <Portfolio stocks={stockDetails} selected={selected} setSelected={setSelected} weights={weights} setWeights={setWeights} startDate={startDate} setStartDate={setStartDate} endDate={endDate} setEndDate={setEndDate} onAnalyze={runRisk} busy={busy} />;
      case "risk": return riskResult?.metrics ? <Risk result={riskResult} navigate={navigate} /> : <section className="content-card"><p>Run a portfolio analysis to view risk metrics.</p><button className="button button-primary" onClick={() => navigate("builder")}>Build a portfolio</button></section>;
      case "forecast": return <Forecast stocks={stockDetails} symbol={forecastSymbol} setSymbol={setForecastSymbol} model={forecastModel} setModel={setForecastModel} onRun={runLiveForecast} result={forecastResult?.forecasted_annualized_volatility !== undefined ? forecastResult : null} busy={busy} />;
      case "optimization": return <Optimization stocks={stockDetails} selected={optimizationSymbols} setSelected={setOptimizationSymbols} strategy={strategy} setStrategy={setStrategy} startDate={startDate} setStartDate={setStartDate} endDate={endDate} setEndDate={setEndDate} onRun={runLiveOptimization} result={optimizationResult?.strategies ? optimizationResult : null} busy={busy} />;
      case "research": return <Research experiments={experiments} models={models} />;
      case "architecture": return <Architecture health={apiHealth} stocks={stocks} />;
      case "about": return <About />;
      default: return <Overview stocks={stockDetails} navigate={navigate} />;
    }
  }

  if (mode === "connecting") return <div className="connection-screen"><span className="eyebrow">FINCLOUD</span><h1>Connecting to AWS services…</h1><p>Verifying the API and private research dataset.</p></div>;
  if (mode === "error") return <div className="connection-screen"><span className="eyebrow">AWS CONNECTION UNAVAILABLE</span><h1>FinCloud could not connect to its research API.</h1><p>{apiProblem}</p><button className="button button-primary" onClick={() => setMode("demo")}>Continue in clearly labeled local demo mode</button><button className="button" onClick={connect}>Retry AWS connection</button></div>;

  const nav = mode === "live" ? NAV_ITEMS : DEMO_NAV;
  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#overview" onClick={(event) => { event.preventDefault(); setPage("dashboard"); }}><span className="brand-mark">F</span><span>fincloud<small>INVESTMENT INTELLIGENCE</small></span></a>
      <div className="workspace-label">{mode === "live" ? "RESEARCH WORKSPACE" : "LOCAL DEMO WORKSPACE"}</div>
      <nav className="primary-nav" aria-label="Main navigation">{nav.map(([id, label, icon]) => <button key={id} className={`nav-item ${page === id ? "nav-active" : ""}`} onClick={() => mode === "live" ? navigate(id) : setPage(id)}><span className="nav-icon">{icon}</span><span>{label}</span></button>)}</nav>
      <div className="sidebar-spacer" />
      <div className="connection-card"><span className="connection-dot" /><div><strong>{mode === "live" ? "AWS connected" : "Local demo mode"}</strong><small>{mode === "live" ? "Lambda · private S3" : "Simulated data only"}</small></div></div>
      <div className="sidebar-footer"><span className="avatar">FC</span><div><strong>FinCloud</strong><small>{mode === "live" ? "AWS-backed research" : "Demo fallback"}</small></div></div>
    </aside>
    <main className="main-area">
      <header className="topbar"><div className="breadcrumb"><span>FinCloud</span><i>/</i><strong>{NAV_ITEMS.find(([id]) => id === page)?.[1]}</strong></div><div className="topbar-actions"><span className="environment-pill"><i />{mode === "live" ? "AWS CONNECTED" : "DEMO MODE"}</span><span className="topbar-date">{mode === "live" ? "HISTORICAL DATA" : "SIMULATED DATA"}</span></div></header>
      {apiProblem && <div className="api-error" role="alert">{apiProblem}</div>}
      <div className="page-container">{mode === "live" ? livePageContent() : <>{demoPageContent()}<p className="demo-disclaimer">{DEMO_DISCLAIMER}</p><button className="button" onClick={connect}>Reconnect to AWS</button></>}</div>
      <footer className="app-footer"><span>FinCloud · {mode === "live" ? "AWS-backed research" : "Local demonstration fallback"}</span><span>{mode === "live" ? `AWS ${apiHealth?.aws_region} · historical dataset` : "Simulated calculations; no backend connection"}</span></footer>
    </main>
  </div>;
}
