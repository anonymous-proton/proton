import { asNumber, fmt, fmtPercent } from "../lib/format";

export type DetailTone = "neutral" | "success" | "warning" | "danger";

export function KeyValueGrid({ items }: { items: Array<{ label: string; value: unknown }> }) {
  return (
    <div className="kv-grid">
      {items.map((item) => (
        <div key={item.label} className="kv-item">
          <span className="kv-label">{item.label}</span>
          <span className="kv-value">{fmt(item.value)}</span>
        </div>
      ))}
    </div>
  );
}

export function JsonDetails({ title, value }: { title: string; value: Record<string, unknown> }) {
  return (
    <details className="json-details">
      <summary>{title}</summary>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </details>
  );
}

export function HealthCard({ title, value, detail }: { title: string; value: string; detail?: string }) {
  return (
    <article className="health-card">
      <span className="health-card__title">{title}</span>
      <strong className="health-card__value">{value}</strong>
      {detail ? <span className="health-card__detail">{detail}</span> : null}
    </article>
  );
}

export function DetailMetric({ label, value, tone = "neutral" }: { label: string; value: string; tone?: DetailTone }) {
  return (
    <article className={`detail-metric detail-metric--${tone}`}>
      <span className="detail-metric__label">{label}</span>
      <strong className="detail-metric__value">{value}</strong>
    </article>
  );
}

export function ReadinessCard({ label, value }: { label: string; value: unknown }) {
  const numeric = Math.max(0, Math.min(1, asNumber(value) ?? 0));
  return (
    <article className="readiness-card">
      <div className="readiness-card__header">
        <span>{label}</span>
        <strong>{fmtPercent(value)}</strong>
      </div>
      <div className="readiness-card__bar">
        <div className="readiness-card__fill" style={{ width: `${numeric * 100}%` }} />
      </div>
    </article>
  );
}
