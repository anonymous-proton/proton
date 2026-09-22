import { fmt, sourceSummaryLine } from "../lib/format";
import type { CampaignSummaryRow, UnassignedSummary } from "../types";
import { Badge } from "./Badge";

interface CampaignQueuePanelProps {
  visibleCampaigns: CampaignSummaryRow[];
  totalCampaigns: number;
  hiddenCampaignCount: number;
  selectedCampaignId: string;
  showTerminalCampaigns: boolean;
  campaignsLoading: boolean;
  hasCampaignsPayload: boolean;
  campaignSummary: Record<string, number>;
  unassigned: UnassignedSummary | undefined;
  onToggleShowTerminalCampaigns: (next: boolean) => void;
  onSelectCampaign: (campaignId: string) => void;
}

export function CampaignQueuePanel({ visibleCampaigns, totalCampaigns, hiddenCampaignCount, selectedCampaignId, showTerminalCampaigns, campaignsLoading, hasCampaignsPayload, campaignSummary, unassigned, onToggleShowTerminalCampaigns, onSelectCampaign }: CampaignQueuePanelProps) {
  return (
    <section className="panel panel--compact">
      <div className="section-header">
        <div>
          <h2>Campaign Queue</h2>
          <p className="small-muted">Click a campaign row to filter Recent Runs and open campaign detail.</p>
        </div>
        <div className="section-actions">
          <div className="small-muted">
            waiting={fmt(campaignSummary.waiting)} active=
            {fmt(campaignSummary.active)} completed=
            {fmt(campaignSummary.completed)} failed=
            {fmt(campaignSummary.failed)}
          </div>
          <label className="toggle-control">
            <input type="checkbox" checked={showTerminalCampaigns} onChange={(event) => onToggleShowTerminalCampaigns(event.target.checked)} />
            <span>show terminal campaigns</span>
          </label>
        </div>
      </div>

      <div className="meta-row">
        <span>campaigns: {visibleCampaigns.length}</span>
        <span>total: {totalCampaigns}</span>
        {hiddenCampaignCount > 0 ? <span>hidden terminal: {hiddenCampaignCount}</span> : null}
        <span>unassigned_rows: {fmt(unassigned?.rows)}</span>
      </div>

      <div className="table-wrap table-wrap--bounded table-wrap--queue">
        <table className="compact-table">
          <thead>
            <tr>
              <th>campaign</th>
              <th>owner</th>
              <th>status</th>
              <th>tasks</th>
            </tr>
          </thead>
          <tbody>
            {visibleCampaigns.map((row) => {
              const counts = row.task_counts || {
                submitted: 0,
                running: 0,
                succeeded: 0,
                failed: 0,
                cancelled: 0,
                observed: 0,
                total: 0,
              };
              const selected = row.campaign_id === selectedCampaignId;
              return (
                <tr key={row.campaign_id} className={selected ? "is-selected" : ""} onClick={() => onSelectCampaign(selected ? "" : row.campaign_id)}>
                  <td>
                    <div className="cell-stack">
                      <strong className="cell-primary">{row.campaign_id}</strong>
                      <span className="small-muted">{row.run_name || "-"}</span>
                      <span className="small-muted">{sourceSummaryLine(row.gateway_summary ?? [])}</span>
                      <span className="small-muted">{row.last_seen_at || "-"}</span>
                    </div>
                  </td>
                  <td>{row.submitter || "-"}</td>
                  <td>
                    <Badge value={row.status} />
                  </td>
                  <td className="small-muted">
                    obs={counts.observed} run={counts.running} ok=
                    {counts.succeeded} fail={counts.failed}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {campaignsLoading && !hasCampaignsPayload ? <p className="small-muted">Loading campaign queue...</p> : null}
      {visibleCampaigns.length === 0 ? <p className="small-muted">No campaigns matched the current terminal toggle and source filters.</p> : null}
    </section>
  );
}
