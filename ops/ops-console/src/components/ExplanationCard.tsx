import type { ReactNode } from "react";

import { fmtSignedWithUnit, fmtWithUnit } from "../lib/format";
import { badgeLabel, badgeTone } from "../lib/status";
import type { TargetExplanationDetail, TargetExplanationSummary } from "../types";
import { Badge } from "./Badge";
import { DetailMetric, type DetailTone, KeyValueGrid } from "./DetailPrimitives";

function explanationTone(status: string): DetailTone {
  const tone = badgeTone(status);
  if (tone === "success" || tone === "warning" || tone === "danger") {
    return tone;
  }
  return "neutral";
}

export function targetDetailLine(target: TargetExplanationSummary): string {
  if (target.status === "abstained") {
    return target.abstain_reason || "no prediction was made for this target";
  }
  if (target.status === "telemetry_missing") {
    return "prediction exists, but actual telemetry is missing";
  }
  return `upper ${fmtWithUnit(target.upper, target.unit)} / actual ${fmtWithUnit(target.actual, target.unit)} / delta ${fmtSignedWithUnit(target.delta_vs_upper, target.unit)}`;
}

export function TargetSummaryCell({ target }: { target: TargetExplanationSummary }) {
  return (
    <div className="target-summary">
      <div className="badge-row">
        <Badge value={target.status} />
        {target.status === "abstained" && target.abstain_reason_code ? <Badge value={target.abstain_reason_code} /> : <Badge value={target.support_level} />}
      </div>
      <span className="small-muted">{targetDetailLine(target)}</span>
      <span className="tiny-note">
        {target.effective_support} rows via {badgeLabel(target.fallback_level)} fallback
        {target.scope ? `, scope ${target.scope}` : ""}
      </span>
    </div>
  );
}

export function CompactExplanation({ explanation }: { explanation: TargetExplanationDetail }) {
  return (
    <div className="drawer-subsection">
      <div className="section-header">
        <h4>{explanation.title}</h4>
        <div className="badge-row">
          <Badge value={explanation.status} />
          <Badge value={explanation.support_level} />
        </div>
      </div>
      <p className="small-muted">{explanation.summary}</p>
      <KeyValueGrid
        items={[
          {
            label: "center",
            value: fmtWithUnit(explanation.center, explanation.unit),
          },
          {
            label: "upper",
            value: fmtWithUnit(explanation.upper, explanation.unit),
          },
          {
            label: "actual",
            value: fmtWithUnit(explanation.actual, explanation.unit),
          },
          {
            label: "delta_vs_upper",
            value: fmtSignedWithUnit(explanation.delta_vs_upper, explanation.unit),
          },
          { label: "fallback_level", value: explanation.fallback_level },
          {
            label: "selected_rows",
            value: explanation.evidence.n_selected_rows,
          },
        ]}
      />
    </div>
  );
}

export function ExplanationCard({ explanation, children }: { explanation: TargetExplanationDetail; children?: ReactNode }) {
  const tone = explanationTone(explanation.status);

  return (
    <section className="drawer-section explanation-card">
      <div className="section-header">
        <div>
          <h3>{explanation.title}</h3>
          <p className="small-muted">{explanation.summary}</p>
        </div>
        <div className="badge-row">
          <Badge value={explanation.status} />
          {explanation.status === "abstained" && explanation.abstain_reason_code ? <Badge value={explanation.abstain_reason_code} /> : <Badge value={explanation.support_level} />}
        </div>
      </div>

      <div className="detail-metric-grid">
        <DetailMetric label="center" value={fmtWithUnit(explanation.center, explanation.unit)} />
        <DetailMetric label="upper" value={fmtWithUnit(explanation.upper, explanation.unit)} tone={tone} />
        <DetailMetric label="actual" value={fmtWithUnit(explanation.actual, explanation.unit)} />
        <DetailMetric label="delta vs upper" value={fmtSignedWithUnit(explanation.delta_vs_upper, explanation.unit)} tone={tone} />
      </div>

      <KeyValueGrid
        items={[
          { label: "support_level", value: explanation.support_level },
          { label: "fallback_level", value: explanation.fallback_level },
          { label: "effective_support", value: explanation.effective_support },
          { label: "scope", value: explanation.scope || "-" },
          {
            label: "guard_margin",
            value: fmtWithUnit(explanation.guard_margin, explanation.unit),
          },
          {
            label: "campaign_correction",
            value: explanation.campaign_correction_total === null ? "-" : fmtSignedWithUnit(explanation.campaign_correction_total, explanation.unit),
          },
        ]}
      />

      {explanation.abstain_reason ? <div className="explanation-note">{explanation.abstain_reason}</div> : null}

      <div className="subsection-grid">
        <div className="drawer-subsection">
          <h4>Evidence</h4>
          <KeyValueGrid
            items={[
              {
                label: "history_rows",
                value: explanation.evidence.n_history_rows,
              },
              { label: "exact_rows", value: explanation.evidence.n_exact_rows },
              {
                label: "nearby_rows",
                value: explanation.evidence.n_nearby_rows,
              },
              {
                label: "coarse_rows",
                value: explanation.evidence.n_coarse_rows,
              },
              {
                label: "selected_rows",
                value: explanation.evidence.n_selected_rows,
              },
              {
                label: "correction_rows",
                value: explanation.evidence.n_campaign_correction_rows,
              },
            ]}
          />
        </div>
        <div className="drawer-subsection">
          <h4>Guard Factors</h4>
          <KeyValueGrid
            items={[
              {
                label: "base",
                value: fmtWithUnit(explanation.guard_components.base, explanation.unit),
              },
              {
                label: "support",
                value: fmtWithUnit(explanation.guard_components.support, explanation.unit),
              },
              {
                label: "fallback",
                value: fmtWithUnit(explanation.guard_components.fallback, explanation.unit),
              },
              {
                label: "safety",
                value: fmtWithUnit(explanation.guard_components.safety, explanation.unit),
              },
              {
                label: "total",
                value: fmtWithUnit(explanation.guard_components.total, explanation.unit),
              },
            ]}
          />
        </div>
      </div>

      {explanation.reason_codes.length > 0 ? (
        <div className="cell-stack">
          <span className="kv-label">scheduler reason codes</span>
          <div className="badge-row">
            {explanation.reason_codes.map((reason) => (
              <span key={reason} className="chip">
                {reason}
              </span>
            ))}
          </div>
        </div>
      ) : null}

      {children}
    </section>
  );
}
