import { useEffect, useState } from "react";

import { fetchCampaignDetail, fetchCampaigns, fetchOverview, fetchRecentRuns, fetchRunDetail, fetchSignals, type SignalsPayload } from "./api";
import { CampaignDetailPanel } from "./components/CampaignDetailPanel";
import { CampaignQueuePanel } from "./components/CampaignQueuePanel";
import { FleetPanel } from "./components/FleetPanel";
import { RecentRunsSection } from "./components/RecentRunsSection";
import { RunDetailDrawer } from "./components/RunDetailDrawer";
import { SignalPanel } from "./components/SignalPanel";
import { SourceFilterMenu } from "./components/SourceFilterMenu";
import { usePollingResource } from "./hooks/usePollingResource";
import { asRecord } from "./lib/format";
import { createEmptySourceFilters, sourceFilterCount, toggleToken, type SourceFilterKey } from "./lib/sourceFilters";
import { isTerminalCampaignStatus } from "./lib/status";
import type { CampaignFilters, CampaignSummaryRow, GatewayAvailableFilters, OverviewPayload } from "./types";

const POLL_MS = 2000;
const RECENT_RUN_LIMIT = 40;

export default function App() {
  const [selectedCampaignId, setSelectedCampaignId] = useState("");
  const [selectedComponent, setSelectedComponent] = useState("");
  const [selectedRunKey, setSelectedRunKey] = useState("");
  const [sourceFilters, setSourceFilters] = useState<CampaignFilters>(createEmptySourceFilters);
  const [sourceDrawerOpen, setSourceDrawerOpen] = useState(false);
  const [showTerminalCampaigns, setShowTerminalCampaigns] = useState(false);

  const overviewState = usePollingResource<OverviewPayload>({
    load: fetchOverview,
    pollMs: POLL_MS,
  });
  const campaignsState = usePollingResource({
    load: () => fetchCampaigns(sourceFilters),
    pollMs: POLL_MS,
    deps: [sourceFilters],
  });
  const campaignDetailState = usePollingResource({
    enabled: Boolean(selectedCampaignId),
    load: () => fetchCampaignDetail(selectedCampaignId, sourceFilters),
    pollMs: POLL_MS,
    deps: [selectedCampaignId, sourceFilters],
  });
  const recentRunsState = usePollingResource({
    load: () =>
      fetchRecentRuns({
        limit: RECENT_RUN_LIMIT,
        campaign_id: selectedCampaignId,
        component: selectedComponent,
        include_gateway_instance_id: sourceFilters.include_gateway_instance_id,
        exclude_gateway_instance_id: sourceFilters.exclude_gateway_instance_id,
        include_gateway_git_commit: sourceFilters.include_gateway_git_commit,
        exclude_gateway_git_commit: sourceFilters.exclude_gateway_git_commit,
      }),
    pollMs: POLL_MS,
    deps: [selectedCampaignId, selectedComponent, sourceFilters],
  });
  const runDetailState = usePollingResource({
    enabled: Boolean(selectedRunKey),
    load: () => fetchRunDetail(selectedRunKey),
    pollMs: POLL_MS,
    deps: [selectedRunKey],
  });
  const signalsState = usePollingResource<SignalsPayload>({
    load: fetchSignals,
    pollMs: POLL_MS * 2,
  });

  const overview = overviewState.data;
  const campaignsPayload = campaignsState.data;
  const campaignDetail = campaignDetailState.data;
  const recentRuns = recentRunsState.data;
  const runDetail = runDetailState.data;

  const campaigns: CampaignSummaryRow[] = campaignsPayload?.campaigns ?? [];
  const visibleCampaigns = campaigns.filter((row) => showTerminalCampaigns || !isTerminalCampaignStatus(row.status));
  const hiddenCampaignCount = campaigns.length - visibleCampaigns.length;
  const fleetQueue = asRecord(overview?.fleet?.queue);
  const queueByComponent = Array.isArray(fleetQueue.by_component) ? (fleetQueue.by_component as Array<Record<string, unknown>>) : [];
  const campaignSummary = campaignsPayload?.summary ?? {
    waiting: 0,
    active: 0,
    completed: 0,
    failed: 0,
    unknown: 0,
  };
  const availableGatewayFilters: GatewayAvailableFilters = campaignsPayload?.available_gateway_filters ?? {
    instances: [],
    commits: [],
  };
  const runs = recentRuns?.runs ?? [];
  const activeSourceFilterCount = sourceFilterCount(sourceFilters);

  const toggleSourceFilter = (key: SourceFilterKey, oppositeKey: SourceFilterKey, token: string) => {
    setSourceFilters((current) => {
      const next: Record<SourceFilterKey, string[]> = {
        include_gateway_instance_id: [...(current.include_gateway_instance_id ?? [])],
        exclude_gateway_instance_id: [...(current.exclude_gateway_instance_id ?? [])],
        include_gateway_git_commit: [...(current.include_gateway_git_commit ?? [])],
        exclude_gateway_git_commit: [...(current.exclude_gateway_git_commit ?? [])],
      };
      next[key] = toggleToken(next[key], token);
      next[oppositeKey] = next[oppositeKey].filter((item) => item !== token);
      return next;
    });
  };

  const selectAllSourceFilters = (key: SourceFilterKey, oppositeKey: SourceFilterKey, tokens: string[]) => {
    setSourceFilters((current) => {
      const next: Record<SourceFilterKey, string[]> = {
        include_gateway_instance_id: [...(current.include_gateway_instance_id ?? [])],
        exclude_gateway_instance_id: [...(current.exclude_gateway_instance_id ?? [])],
        include_gateway_git_commit: [...(current.include_gateway_git_commit ?? [])],
        exclude_gateway_git_commit: [...(current.exclude_gateway_git_commit ?? [])],
      };
      next[key] = Array.from(new Set(tokens));
      next[oppositeKey] = [];
      return next;
    });
  };

  const clearSourceFilters = (key: SourceFilterKey, oppositeKey: SourceFilterKey) => {
    setSourceFilters((current) => {
      const next: Record<SourceFilterKey, string[]> = {
        include_gateway_instance_id: [...(current.include_gateway_instance_id ?? [])],
        exclude_gateway_instance_id: [...(current.exclude_gateway_instance_id ?? [])],
        include_gateway_git_commit: [...(current.include_gateway_git_commit ?? [])],
        exclude_gateway_git_commit: [...(current.exclude_gateway_git_commit ?? [])],
      };
      next[key] = [];
      next[oppositeKey] = [];
      return next;
    });
  };

  useEffect(() => {
    if (!selectedCampaignId) {
      return;
    }
    const exists = visibleCampaigns.some((row) => row.campaign_id === selectedCampaignId);
    if (!exists) {
      setSelectedCampaignId("");
    }
  }, [selectedCampaignId, visibleCampaigns]);

  useEffect(() => {
    if (!selectedComponent) {
      return;
    }
    const exists = queueByComponent.some((row) => String(row.component || "") === selectedComponent);
    if (!exists) {
      setSelectedComponent("");
    }
  }, [queueByComponent, selectedComponent]);

  useEffect(() => {
    if (!selectedRunKey) {
      return;
    }
    const exists = runs.some((row) => row.run_key === selectedRunKey);
    if (!exists) {
      setSelectedRunKey("");
    }
  }, [runs, selectedRunKey]);

  if (overviewState.loading && !overview) {
    return (
      <main className="page">
        <p>Loading campaign operations overview...</p>
      </main>
    );
  }

  return (
    <>
      {selectedRunKey ? <div className="drawer-backdrop" onClick={() => setSelectedRunKey("")} aria-hidden="true" /> : null}
      <main className={`page ${selectedRunKey ? "page--drawer-open" : ""}`}>
        <header className="hero">
          <div>
            <h1>Campaign-First Operations Console</h1>
            <p className="small-muted">The run table now explains whether runtime and active memory predicted, abstained, or missed the guard, and the drawer keeps the target-level evidence ahead of raw JSON.</p>
          </div>
          <div className="hero-controls">
            {overview ? (
              <div className="meta-row">
                <span className={overview.meta.stale ? "stale-badge stale" : "stale-badge fresh"}>{overview.meta.stale ? "stale" : "fresh"}</span>
                <span>generated_at: {overview.meta.generated_at}</span>
                <span>data_age_seconds: {overview.meta.data_age_seconds.toFixed(2)}</span>
              </div>
            ) : null}
            <SourceFilterMenu
              sourceDrawerOpen={sourceDrawerOpen}
              activeSourceFilterCount={activeSourceFilterCount}
              sourceFilters={sourceFilters}
              availableGatewayFilters={availableGatewayFilters}
              onToggleDrawer={() => setSourceDrawerOpen((current) => !current)}
              onResetFilters={() => setSourceFilters(createEmptySourceFilters())}
              onToggleSourceFilter={toggleSourceFilter}
              onSelectAllSourceFilters={selectAllSourceFilters}
              onClearSourceFilters={clearSourceFilters}
            />
          </div>
        </header>

        {overviewState.error ? <div className="error-banner">{overviewState.error}</div> : null}
        {campaignsState.error ? <div className="error-banner">{campaignsState.error}</div> : null}

        <RecentRunsSection
          summary={recentRuns?.summary}
          recentRuns={recentRuns}
          runs={runs}
          runsError={recentRunsState.error}
          runsLoading={recentRunsState.loading}
          selectedCampaignId={selectedCampaignId}
          selectedComponent={selectedComponent}
          selectedRunKey={selectedRunKey}
          onClearSelectedCampaign={() => setSelectedCampaignId("")}
          onClearSelectedComponent={() => setSelectedComponent("")}
          onClearFilters={() => {
            setSelectedCampaignId("");
            setSelectedComponent("");
          }}
          onSelectRunKey={setSelectedRunKey}
        />

        <div className="content-grid">
          <div className="sidebar-stack">
            <CampaignQueuePanel visibleCampaigns={visibleCampaigns} totalCampaigns={campaigns.length} hiddenCampaignCount={hiddenCampaignCount} selectedCampaignId={selectedCampaignId} showTerminalCampaigns={showTerminalCampaigns} campaignsLoading={campaignsState.loading} hasCampaignsPayload={Boolean(campaignsPayload)} campaignSummary={campaignSummary} unassigned={campaignsPayload?.unassigned} onToggleShowTerminalCampaigns={setShowTerminalCampaigns} onSelectCampaign={setSelectedCampaignId} />

            <FleetPanel queueByComponent={queueByComponent} selectedComponent={selectedComponent} totalQueueDepth={fleetQueue.total_queue_depth} totalInflight={fleetQueue.total_inflight} onSelectComponent={setSelectedComponent} />
          </div>

          <CampaignDetailPanel selectedCampaignId={selectedCampaignId} campaignDetail={campaignDetail} campaignError={campaignDetailState.error} campaignLoading={campaignDetailState.loading} />
        </div>

        <SignalPanel signals={signalsState.data} loading={signalsState.loading} error={signalsState.error} />
      </main>

      <RunDetailDrawer selectedRunKey={selectedRunKey} runDetail={runDetail} detailLoading={runDetailState.loading} detailError={runDetailState.error} onClose={() => setSelectedRunKey("")} />
    </>
  );
}
