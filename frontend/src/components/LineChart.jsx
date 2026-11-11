export default function LineChart({ points = [], label = "Portfolio growth" }) {
  const valid = points.filter((point) => Number.isFinite(point.return));
  if (valid.length < 2) {
    return <div className="chart-empty">At least two valid observations are needed for this chart.</div>;
  }
  let wealth = 1;
  const values = valid.map((point) => {
    wealth *= 1 + point.return;
    return { ...point, wealth };
  });
  const min = Math.min(...values.map((point) => point.wealth));
  const max = Math.max(...values.map((point) => point.wealth));
  const span = max - min || 1;
  const path = values
    .map((point, index) => {
      const x = (index / (values.length - 1)) * 1000;
      const y = 185 - ((point.wealth - min) / span) * 160;
      return `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <div className="chart-wrap" role="img" aria-label={`${label} from ${values[0].date} to ${values.at(-1).date}`}>
      <div className="chart-head"><span>{label}</span><span>{values.length.toLocaleString()} daily observations</span></div>
      <svg viewBox="0 0 1000 220" preserveAspectRatio="none">
        <defs>
          <linearGradient id="chartFill" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#52dda1" stopOpacity=".24" />
            <stop offset="100%" stopColor="#52dda1" stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={`${path} L1000,220 L0,220 Z`} fill="url(#chartFill)" />
        <path d={path} fill="none" stroke="#52dda1" strokeWidth="3" vectorEffect="non-scaling-stroke" />
      </svg>
      <div className="chart-axis"><span>{values[0].date}</span><span>Compounded value (starting at 1.00)</span><span>{values.at(-1).date}</span></div>
    </div>
  );
}
