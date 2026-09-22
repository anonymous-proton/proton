import { formatDistribution } from "../lib/format";
import { badgeLabel } from "../lib/status";
import type { EstimatorRunSummary, RecentRunRow, RecentRunsPayload } from "../types";
import { Badge } from "./Badge";
import { HealthCard } from "./DetailPrimitives";
import { TargetSummaryCell } from "./ExplanationCard";
import { Glossary } from "./Glossary";

interface RecentRunsSectionProps {
  summary?: EstimatorRunSummary | null;
  recentRuns: RecentRunsPayload | null;
  runs: RecentRunRow[];
  runsError: string;
  runsLoading: boolean;
  selectedCampaignId: string;
  selectedComponent: string;
  selectedRunKey: string;
  onClearSelectedCampaign: () => void;
  onClearSelectedComponent: () => void;
  onClearFilters: () => void;
  onSelectRunKey: (runKey: string) => void;
}

export function RecentRunsSection({ summary, recentRuns, runs, runsError, runsLoading, selectedCampaignId, selectedComponent, selectedRunKey, onClearSelectedCampaign, onClearSelectedComponent, onClearFilters, onSelectRunKey }: RecentRunsSectionProps) {
  return (
    <section className="panel">
      {summary ? (
        <section className="health-strip">
          <HealthCard title="last 1h runs" value={String(summary.last_1h_runs)} />
          <HealthCard title="runtime support" value={formatDistribution(summary.runtime_support_distribution)} detail="exact / nearby / coarse / none" />
          <HealthCard title="active-memory support" value={formatDistribution(summary.active_memory_support_distribution)} detail="exact / nearby / coarse / none" />
          <HealthCard title="guard misses" value={`rt ${summary.runtime_violation_count} / mem ${summary.active_memory_violation_count}`} />
          <HealthCard title="abstentions" value={`rt ${summary.runtime_abstention_count} / mem ${summary.active_memory_abstention_count} / base ${summary.resident_baseline_abstention_count}`} />
          <HealthCard title="telemetry / retries" value={`missing ${summary.telemetry_missing_count}`} detail={`post-selection retries ${summary.post_selection_retry_count}, campaign correction ${summary.campaign_correction_applied_count}`} />
        </section>
      ) : null}

      <div className="section-header">
        <div>
          <h2>Recent Runs</h2>
          <p className="small-muted">newest first, with target-specific explanations instead of compressed support and fallback badges</p>
        </div>
        <div className="section-actions">
          {selectedCampaignId ? (
            <button className="chip chip-button" onClick={onClearSelectedCampaign} type="button">
              campaign: {selectedCampaignId} x
            </button>
          ) : null}
          {selectedComponent ? (
            <button className="chip chip-button" onClick={onClearSelectedComponent} type="button">
              component: {selectedComponent} x
            </button>
          ) : null}
          {selectedCampaignId || selectedComponent ? (
            <button className="button-secondary" onClick={onClearFilters} type="button">
              clear filters
            </button>
          ) : null}
        </div>
      </div>

      {runsError ? <div className="error-banner">{runsError}</div> : null}

      <div className="meta-row">
        <span>limit: {recentRuns?.limit ?? 0}</span>
        <span>total: {recentRuns?.total ?? 0}</span>
        <span>generated_at: {recentRuns?.generated_at ?? "-"}</span>
      </div>

      <Glossary />

      <div className="table-wrap table-wrap--bounded table-wrap--runs">
        <table className="runs-table">
          <thead>
            <tr>
              <th>run</th>
              <th>routing context</th>
              <th>runtime</th>
              <th>active memory</th>
              <th>takeaway</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((row) => {
              const selected = row.run_key === selectedRunKey;
              return (
                <tr key={row.run_key} className={selected ? "is-selected" : ""} onClick={() => onSelectRunKey(row.run_key)}>
                  <td>
                    <div className="cell-stack">
                      <span>{row.time || "-"}</span>
                      <strong className="cell-primary">{row.stage_id}</strong>
                      <div className="badge-row">
                        <Badge value={row.status} />
                        <Badge value={row.selected_context_applied} />
                      </div>
                      <span className="small-muted">{row.campaign_id}</span>
                      <span className="small-muted">{row.gateway_label}</span>
                      <span className="small-muted mono-cell">{row.run_key}</span>
                    </div>
                  </td>
                  <td>
                    <div className="cell-stack">
                      <strong className="cell-primary mono-cell">{row.worker_id}</strong>
                      <span className="truncate-text mono-cell" title={row.selected_worker_context_summary || "-"}>
                        {row.selected_worker_context_summary || "-"}
                      </span>
                      <span className="tiny-note mono-cell">
                        {row.local_regime_bucket} / {row.co_location_signature}
                      </span>
                    </div>
                  </td>
                  <td>
                    <TargetSummaryCell target={row.runtime} />
                  </td>
                  <td>
                    <TargetSummaryCell target={row.active_memory} />
                  </td>
                  <td>
                    <div className="cell-stack">
                      <span className="cell-primary">{row.takeaway}</span>
                      {row.telemetry_missing ? <Badge value="telemetry_missing" /> : null}
                      <span className="tiny-note">resident baseline: {badgeLabel(row.resident_baseline.status)}</span>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {!runsLoading && runs.length === 0 ? <p className="small-muted">No runs matched the current campaign/component filters.</p> : null}
    </section>
  );
}
