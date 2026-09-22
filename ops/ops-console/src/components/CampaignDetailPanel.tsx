import { asRecord, fmt } from "../lib/format";
import type { CampaignDetailPayload, GatewaySourceSummary } from "../types";
import { Badge } from "./Badge";
import { DetailMetric, ReadinessCard } from "./DetailPrimitives";

interface CampaignDetailPanelProps {
  selectedCampaignId: string;
  campaignDetail: CampaignDetailPayload | null;
  campaignError: string;
  campaignLoading: boolean;
}

function asArrayOfRecords(value: unknown): Array<Record<string, unknown>> {
  return Array.isArray(value) ? (value as Array<Record<string, unknown>>) : [];
}

export function CampaignDetailPanel({ selectedCampaignId, campaignDetail, campaignError, campaignLoading }: CampaignDetailPanelProps) {
  const detailOverview = asRecord(campaignDetail?.overview);
  const readiness = asRecord(campaignDetail?.telemetry_readiness);
  const detailComponents = asArrayOfRecords(campaignDetail?.components);
  const campaignTaskCounts = campaignDetail?.task_counts ?? {
    submitted: 0,
    running: 0,
    succeeded: 0,
    failed: 0,
    cancelled: 0,
    observed: 0,
    total: 0,
  };
  const campaignGatewaySummary: GatewaySourceSummary[] = Array.isArray(campaignDetail?.gateway_summary) ? campaignDetail.gateway_summary : Array.isArray(detailOverview.gateway_summary) ? (detailOverview.gateway_summary as GatewaySourceSummary[]) : [];

  return (
    <section className="panel panel--detail">
      <div className="section-header">
        <div>
          <h2>Campaign Detail</h2>
          <p className="small-muted">selected campaign: {selectedCampaignId || "-"}</p>
        </div>
      </div>

      {campaignError ? <div className="error-banner">{campaignError}</div> : null}
      {campaignLoading ? <p className="small-muted">Loading campaign detail...</p> : null}

      {campaignDetail ? (
        <>
          <div className="campaign-detail-hero">
            <div className="campaign-detail-hero__main">
              <div className="campaign-detail-hero__heading">
                <div>
                  <div className="small-muted">campaign_id</div>
                  <div className="campaign-detail-id">{campaignDetail.campaign_id}</div>
                </div>
                <Badge value={campaignDetail.status || String(detailOverview.status || "unknown")} />
              </div>
              <div className="campaign-detail-hero__meta">
                <span className="chip">run_name: {fmt(campaignDetail.run_name || detailOverview.run_name)}</span>
                <span className="chip">submitter: {fmt(campaignDetail.submitter || detailOverview.submitter)}</span>
                <span className="chip">started: {fmt(campaignDetail.started_at || detailOverview.started_at)}</span>
                <span className="chip">ended: {fmt(campaignDetail.ended_at || detailOverview.ended_at)}</span>
              </div>
            </div>
          </div>

          <div className="campaign-detail-metrics">
            <DetailMetric label="observed" value={String(campaignTaskCounts.observed)} />
            <DetailMetric label="running" value={String(campaignTaskCounts.running)} tone={campaignTaskCounts.running > 0 ? "warning" : "neutral"} />
            <DetailMetric label="succeeded" value={String(campaignTaskCounts.succeeded)} tone={campaignTaskCounts.succeeded > 0 ? "success" : "neutral"} />
            <DetailMetric label="failed" value={String(campaignTaskCounts.failed + campaignTaskCounts.cancelled)} tone={campaignTaskCounts.failed + campaignTaskCounts.cancelled > 0 ? "danger" : "neutral"} />
          </div>

          <div className="campaign-detail-body">
            <div className="campaign-detail-section">
              <div className="section-header">
                <div>
                  <h3>Gateway Sources</h3>
                  <p className="small-muted">Which gateway instance and deploy produced the rows in this filtered campaign view.</p>
                </div>
                <div className="small-muted">sources={campaignGatewaySummary.length}</div>
              </div>
              <div className="badge-row">
                {campaignGatewaySummary.length > 0 ? (
                  campaignGatewaySummary.map((item) => (
                    <span key={`${String(item.instance_id || "source")}-${String(item.git_commit || "commit")}`} className="chip">
                      {String(item.label || "unknown")} ({fmt(item.count, 0)})
                    </span>
                  ))
                ) : (
                  <span className="small-muted">No source summary available.</span>
                )}
              </div>
            </div>

            <div className="campaign-detail-section">
              <div className="section-header">
                <div>
                  <h3>Telemetry Readiness</h3>
                  <p className="small-muted">How complete the succeeded rows are for runtime-facing fields.</p>
                </div>
                <div className="small-muted">succeeded_rows={fmt(readiness.succeeded_rows, 0)}</div>
              </div>
              <div className="readiness-grid">
                <ReadinessCard label="runtime" value={readiness.runtime_present_rate} />
                <ReadinessCard label="peak memory" value={readiness.peak_memory_present_rate} />
                <ReadinessCard label="gpu util" value={readiness.mean_gpu_util_present_rate} />
                <ReadinessCard label="wallclock" value={readiness.wallclock_present_rate} />
              </div>
            </div>

            <div className="campaign-detail-section">
              <div className="section-header">
                <div>
                  <h3>Component Breakdown</h3>
                  <p className="small-muted">Per-component view of what this campaign has actually done.</p>
                </div>
                <div className="small-muted">components={detailComponents.length}</div>
              </div>

              <div className="campaign-component-grid">
                {detailComponents.map((row, idx) => (
                  <article className="campaign-component-card" key={`${String(row.component || "component")}-${idx}`}>
                    <div className="campaign-component-card__header">
                      <strong>{fmt(row.component)}</strong>
                      <span className="small-muted">last_seen {fmt(row.last_seen_at)}</span>
                    </div>
                    <div className="campaign-component-card__stats">
                      <span>observed {fmt(row.observed, 0)}</span>
                      <span>active {fmt(row.active, 0)}</span>
                      <span>succeeded {fmt(row.succeeded, 0)}</span>
                      <span>failed {fmt(row.failed, 0)}</span>
                    </div>
                  </article>
                ))}
              </div>
            </div>
          </div>
        </>
      ) : (
        <p className="small-muted">Select a campaign row to open detail and filter Recent Runs.</p>
      )}
    </section>
  );
}
