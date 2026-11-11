export default function MetricCard({ label, value, detail, tone = "" }) {
  return (
    <article className={`metric-card ${tone}`}>
      <span className="metric-label">{label}</span>
      <strong className="metric-value">{value}</strong>
      {detail && <span className="metric-detail">{detail}</span>}
    </article>
  );
}
