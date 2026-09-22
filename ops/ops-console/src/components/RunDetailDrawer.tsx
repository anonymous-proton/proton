import { useEffect, useState } from "react";

import { ExplanationCard, CompactExplanation } from "./ExplanationCard";
import { Badge } from "./Badge";
import { JsonDetails, KeyValueGrid } from "./DetailPrimitives";
import { fmt, fmtSignedWithUnit, fmtWithUnit } from "../lib/format";
import type { RunDetailPayload } from "../types";

interface RunDetailDrawerProps {
  selectedRunKey: string;
  runDetail: RunDetailPayload | null;
  detailLoading: boolean;
  detailError: string;
  onClose: () => void;
}

function sectionLines(title: string, items: Array<{ label: string; value: unknown }>): string {
  const lines = [title];
  for (const item of items) {
    lines.push(`${item.label}: ${fmt(item.value)}`);
  }
  return lines.join("\n");
}

function explanationLines(title: string, explanation: RunDetailPayload["estimator_explanations"]["runtime"]): string {
  return [
    title,
    `title: ${explanation.title}`,
    `summary: ${explanation.summary}`,
    `status: ${explanation.status}`,
    `support_level: ${explanation.support_level}`,
    `fallback_level: ${explanation.fallback_level}`,
    `effective_support: ${explanation.effective_support}`,
    `scope: ${explanation.scope || "-"}`,
    `center: ${fmtWithUnit(explanation.center, explanation.unit)}`,
    `upper: ${fmtWithUnit(explanation.upper, explanation.unit)}`,
    `actual: ${fmtWithUnit(explanation.actual, explanation.unit)}`,
    `delta_vs_upper: ${fmtSignedWithUnit(explanation.delta_vs_upper, explanation.unit)}`,
    `guard_margin: ${fmtWithUnit(explanation.guard_margin, explanation.unit)}`,
    `campaign_correction_total: ${
      explanation.campaign_correction_total === null
        ? "-"
        : fmtSignedWithUnit(explanation.campaign_correction_total, explanation.unit)
    }`,
    `abstain_reason_code: ${explanation.abstain_reason_code || "-"}`,
    `abstain_reason: ${explanation.abstain_reason || "-"}`,
    `evidence.n_history_rows: ${explanation.evidence.n_history_rows}`,
    `evidence.n_exact_rows: ${explanation.evidence.n_exact_rows}`,
    `evidence.n_nearby_rows: ${explanation.evidence.n_nearby_rows}`,
    `evidence.n_coarse_rows: ${explanation.evidence.n_coarse_rows}`,
    `evidence.n_selected_rows: ${explanation.evidence.n_selected_rows}`,
    `evidence.n_campaign_correction_rows: ${explanation.evidence.n_campaign_correction_rows}`,
    `guard_components.base: ${fmtWithUnit(explanation.guard_components.base, explanation.unit)}`,
    `guard_components.support: ${fmtWithUnit(explanation.guard_components.support, explanation.unit)}`,
    `guard_components.fallback: ${fmtWithUnit(explanation.guard_components.fallback, explanation.unit)}`,
    `guard_components.safety: ${fmtWithUnit(explanation.guard_components.safety, explanation.unit)}`,
    `guard_components.total: ${fmtWithUnit(explanation.guard_components.total, explanation.unit)}`,
    `reason_codes: ${explanation.reason_codes.length > 0 ? explanation.reason_codes.join(", ") : "-"}`,
  ].join("\n");
}

function buildRunDetailClipboardText(runDetail: RunDetailPayload): string {
  return [
    "Run Detail",
    `generated_at: ${runDetail.generated_at}`,
    `run_key: ${runDetail.run_key}`,
    "",
    "Takeaway",
    `headline: ${runDetail.takeaway.headline}`,
    `status: ${runDetail.takeaway.status}`,
    `notes: ${runDetail.takeaway.notes.length > 0 ? runDetail.takeaway.notes.join(" | ") : "-"}`,
    "",
    explanationLines("Estimator / Runtime", runDetail.estimator_explanations.runtime),
    "",
    explanationLines("Estimator / Hot Execution", runDetail.estimator_explanations.hot_execution_time),
    "",
    explanationLines("Estimator / Transition Penalty", runDetail.estimator_explanations.transition_penalty),
    "",
    explanationLines("Estimator / Active Memory", runDetail.estimator_explanations.active_memory),
    "",
    explanationLines("Estimator / Resident Baseline", runDetail.estimator_explanations.resident_baseline),
    "",
    sectionLines("Request / Trace Context", [
      { label: "campaign_id", value: runDetail.request_trace_context.campaign_id },
      { label: "stage_id", value: runDetail.request_trace_context.stage_id },
      { label: "model_id", value: runDetail.request_trace_context.model_id },
      { label: "stage_class", value: runDetail.request_trace_context.stage_class },
      { label: "descriptor_bucket", value: runDetail.request_trace_context.descriptor_bucket },
      { label: "requested_batch_size", value: runDetail.request_trace_context.requested_batch_size },
      { label: "effective_batch_bucket", value: runDetail.request_trace_context.effective_batch_bucket },
    ]),
    "",
    sectionLines("Gateway Source", [
      { label: "gateway_label", value: runDetail.overview.gateway_label },
      { label: "instance_id", value: runDetail.overview.gateway.instance_id },
      { label: "bind_addr", value: runDetail.overview.gateway.bind_addr },
      { label: "git_commit", value: runDetail.overview.gateway.git_commit },
      { label: "started_at", value: runDetail.overview.gateway.started_at },
    ]),
    "",
    sectionLines("Selected Worker Context", [
      { label: "worker_id", value: runDetail.selected_worker_context.worker_id },
      { label: "hardware_software", value: runDetail.selected_worker_context.hardware_software },
      { label: "active_request_count_bucket", value: runDetail.selected_worker_context.active_request_count_bucket },
      { label: "co_location_signature", value: runDetail.selected_worker_context.co_location_signature },
      { label: "residency_state", value: runDetail.selected_worker_context.residency_state },
    ]),
    "",
    sectionLines("Actual Telemetry", [
      { label: "actual_hot_execution_sec", value: runDetail.actual_telemetry.actual_hot_execution_sec },
      { label: "actual_transition_penalty_sec", value: runDetail.actual_telemetry.actual_transition_penalty_sec },
      { label: "actual_total_runtime_sec", value: runDetail.actual_telemetry.actual_total_runtime_sec },
      { label: "actual_active_memory_mib", value: runDetail.actual_telemetry.actual_active_memory_mib },
      { label: "dispatch_time_resident_baseline_mib", value: runDetail.actual_telemetry.dispatch_time_resident_baseline_mib },
      { label: "dispatch_time_resident_baseline_state", value: runDetail.actual_telemetry.dispatch_time_resident_baseline_state },
      { label: "failure_outcome", value: runDetail.actual_telemetry.failure_outcome || "-" },
      { label: "runtime_violation", value: runDetail.actual_telemetry.runtime_violation },
      { label: "memory_violation", value: runDetail.actual_telemetry.memory_violation },
      { label: "telemetry_missing", value: runDetail.actual_telemetry.telemetry_missing },
    ]),
    "",
    "Raw JSON / run_record",
    JSON.stringify(runDetail.raw.run_record, null, 2),
    "",
    "Raw JSON / scheduler_decision",
    JSON.stringify(runDetail.raw.scheduler_decision, null, 2),
    "",
    "Raw JSON / axes",
    JSON.stringify(runDetail.raw.axes, null, 2),
  ].join("\n");
}

async function copyTextToClipboard(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textArea = document.createElement("textarea");
  textArea.value = text;
  textArea.setAttribute("readonly", "true");
  textArea.style.position = "fixed";
  textArea.style.opacity = "0";
  document.body.appendChild(textArea);
  textArea.select();
  document.execCommand("copy");
  document.body.removeChild(textArea);
}

export function RunDetailDrawer({ selectedRunKey, runDetail, detailLoading, detailError, onClose }: RunDetailDrawerProps) {
  const [copyState, setCopyState] = useState<"idle" | "done" | "failed">("idle");

  useEffect(() => {
    setCopyState("idle");
  }, [runDetail?.run_key]);

  const handleCopyRunDetail = async () => {
    if (!runDetail) {
      return;
    }
    try {
      await copyTextToClipboard(buildRunDetailClipboardText(runDetail));
      setCopyState("done");
    } catch {
      setCopyState("failed");
    }
  };

  return (
    <aside className={`drawer ${selectedRunKey ? "drawer--open" : ""}`}>
      <div className="drawer__header">
        <div>
          <h2>Run Detail</h2>
          <p className="small-muted">{selectedRunKey || "Select a run row to inspect one run at a time."}</p>
        </div>
        <div className="drawer__actions">
          {runDetail ? (
            <button className="button-secondary" onClick={() => void handleCopyRunDetail()} type="button">
              copy to clipboard
            </button>
          ) : null}
          <button className="button-secondary" onClick={onClose} type="button">
            close
          </button>
        </div>
      </div>

      {copyState === "done" ? <div className="copy-banner">Run detail copied as text.</div> : null}
      {copyState === "failed" ? <div className="copy-banner copy-banner--error">Clipboard copy failed.</div> : null}
      {detailError ? <div className="error-banner">{detailError}</div> : null}
      {detailLoading && !runDetail ? <p className="small-muted">Loading run detail...</p> : null}

      {runDetail ? (
        <div className="drawer__body">
          <section className="drawer-section takeaway-panel">
            <div className="section-header">
              <div>
                <h3>Takeaway</h3>
                <p className="takeaway-headline">{runDetail.takeaway.headline}</p>
              </div>
              <div className="badge-row">
                <Badge value={runDetail.takeaway.status} />
                <Badge value={runDetail.overview.status} />
                <Badge value={runDetail.overview.selected_context_applied} />
              </div>
            </div>
            {runDetail.takeaway.notes.length > 0 ? (
              <div className="note-list">
                {runDetail.takeaway.notes.map((note) => (
                  <div key={note} className="explanation-note">
                    {note}
                  </div>
                ))}
              </div>
            ) : null}
          </section>

          <ExplanationCard explanation={runDetail.estimator_explanations.runtime}>
            <details className="drawer-disclosure">
              <summary>Hot execution and transition split</summary>
              <div className="subsection-grid">
                <CompactExplanation explanation={runDetail.estimator_explanations.hot_execution_time} />
                <CompactExplanation explanation={runDetail.estimator_explanations.transition_penalty} />
              </div>
            </details>
          </ExplanationCard>

          <ExplanationCard explanation={runDetail.estimator_explanations.active_memory} />

          <ExplanationCard explanation={runDetail.estimator_explanations.resident_baseline} />

          <section className="drawer-section">
            <h3>Request / Trace Context</h3>
            <KeyValueGrid
              items={[
                {
                  label: "campaign_id",
                  value: runDetail.request_trace_context.campaign_id,
                },
                {
                  label: "stage_id",
                  value: runDetail.request_trace_context.stage_id,
                },
                {
                  label: "model_id",
                  value: runDetail.request_trace_context.model_id,
                },
                {
                  label: "stage_class",
                  value: runDetail.request_trace_context.stage_class,
                },
                {
                  label: "descriptor_bucket",
                  value: runDetail.request_trace_context.descriptor_bucket,
                },
                {
                  label: "requested_batch_size",
                  value: runDetail.request_trace_context.requested_batch_size,
                },
                {
                  label: "effective_batch_bucket",
                  value: runDetail.request_trace_context.effective_batch_bucket,
                },
              ]}
            />
          </section>

          <section className="drawer-section">
            <h3>Gateway Source</h3>
            <KeyValueGrid
              items={[
                {
                  label: "gateway_label",
                  value: runDetail.overview.gateway_label,
                },
                {
                  label: "instance_id",
                  value: runDetail.overview.gateway.instance_id,
                },
                {
                  label: "bind_addr",
                  value: runDetail.overview.gateway.bind_addr,
                },
                {
                  label: "git_commit",
                  value: runDetail.overview.gateway.git_commit,
                },
                {
                  label: "started_at",
                  value: runDetail.overview.gateway.started_at,
                },
              ]}
            />
          </section>

          <section className="drawer-section">
            <h3>Selected Worker Context</h3>
            <KeyValueGrid
              items={[
                {
                  label: "worker_id",
                  value: runDetail.selected_worker_context.worker_id,
                },
                {
                  label: "hardware_software",
                  value: runDetail.selected_worker_context.hardware_software,
                },
                {
                  label: "active_request_count_bucket",
                  value: runDetail.selected_worker_context.active_request_count_bucket,
                },
                {
                  label: "co_location_signature",
                  value: runDetail.selected_worker_context.co_location_signature,
                },
                {
                  label: "residency_state",
                  value: runDetail.selected_worker_context.residency_state,
                },
              ]}
            />
          </section>

          <section className="drawer-section">
            <h3>Actual Telemetry</h3>
            <KeyValueGrid
              items={[
                {
                  label: "actual_hot_execution_sec",
                  value: runDetail.actual_telemetry.actual_hot_execution_sec,
                },
                {
                  label: "actual_transition_penalty_sec",
                  value: runDetail.actual_telemetry.actual_transition_penalty_sec,
                },
                {
                  label: "actual_total_runtime_sec",
                  value: runDetail.actual_telemetry.actual_total_runtime_sec,
                },
                {
                  label: "actual_active_memory_mib",
                  value: runDetail.actual_telemetry.actual_active_memory_mib,
                },
                {
                  label: "dispatch_time_resident_baseline_mib",
                  value: runDetail.actual_telemetry.dispatch_time_resident_baseline_mib,
                },
                {
                  label: "dispatch_time_resident_baseline_state",
                  value: runDetail.actual_telemetry.dispatch_time_resident_baseline_state,
                },
                {
                  label: "failure_outcome",
                  value: runDetail.actual_telemetry.failure_outcome || "-",
                },
                {
                  label: "runtime_violation",
                  value: runDetail.actual_telemetry.runtime_violation,
                },
                {
                  label: "memory_violation",
                  value: runDetail.actual_telemetry.memory_violation,
                },
                {
                  label: "telemetry_missing",
                  value: runDetail.actual_telemetry.telemetry_missing,
                },
              ]}
            />
          </section>

          <section className="drawer-section">
            <h3>Raw JSON</h3>
            <JsonDetails title="run record" value={runDetail.raw.run_record} />
            <JsonDetails title="scheduler decision" value={runDetail.raw.scheduler_decision} />
            <JsonDetails title="axes" value={runDetail.raw.axes} />
          </section>
        </div>
      ) : null}
    </aside>
  );
}
