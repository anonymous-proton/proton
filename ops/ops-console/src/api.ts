import type {
  CampaignDetailPayload,
  CampaignFilters,
  CampaignsPayload,
  OverviewPayload,
  RecentRunFilters,
  RecentRunsPayload,
  RunDetailPayload
} from "./types";

function appendMany(params: URLSearchParams, key: string, values?: string[]) {
  for (const value of values ?? []) {
    const token = String(value || "").trim();
    if (token) {
      params.append(key, token);
    }
  }
}

function buildParams(filters: RecentRunFilters | CampaignFilters): URLSearchParams {
  const params = new URLSearchParams();
  if ("component" in filters && filters.component) {
    params.set("component", filters.component);
  }
  if ("worker_name" in filters && filters.worker_name) {
    params.set("worker_name", filters.worker_name);
  }
  if ("campaign_id" in filters && filters.campaign_id) {
    params.set("campaign_id", filters.campaign_id);
  }
  if ("limit" in filters && filters.limit) {
    params.set("limit", String(filters.limit));
  }
  appendMany(params, "include_gateway_instance_id", filters.include_gateway_instance_id);
  appendMany(params, "exclude_gateway_instance_id", filters.exclude_gateway_instance_id);
  appendMany(params, "include_gateway_git_commit", filters.include_gateway_git_commit);
  appendMany(params, "exclude_gateway_git_commit", filters.exclude_gateway_git_commit);
  return params;
}

async function requestJson<T>(path: string, errorPrefix: string): Promise<T> {
  const res = await fetch(path, {
    headers: {
      Accept: "application/json"
    }
  });
  if (!res.ok) {
    throw new Error(`${errorPrefix}: ${res.status}`);
  }
  return (await res.json()) as T;
}

export async function fetchOverview(): Promise<OverviewPayload> {
  return requestJson<OverviewPayload>("/api/v1/ops/overview", "ops-api overview request failed");
}

export async function fetchCampaigns(filters: CampaignFilters): Promise<CampaignsPayload> {
  const params = buildParams(filters);
  const query = params.toString();
  return requestJson<CampaignsPayload>(
    `/api/v1/ops/campaigns${query ? `?${query}` : ""}`,
    "campaign queue request failed"
  );
}

export async function fetchCampaignDetail(campaignId: string, filters: CampaignFilters): Promise<CampaignDetailPayload> {
  const params = buildParams(filters);
  const query = params.toString();
  return requestJson<CampaignDetailPayload>(
    `/api/v1/ops/campaigns/${encodeURIComponent(campaignId)}${query ? `?${query}` : ""}`,
    "campaign detail request failed"
  );
}

export async function fetchRecentRuns(filters: RecentRunFilters): Promise<RecentRunsPayload> {
  const params = buildParams(filters);
  return requestJson<RecentRunsPayload>(`/api/v1/ops/runs/recent?${params.toString()}`, "recent runs request failed");
}

export async function fetchRunDetail(runKey: string): Promise<RunDetailPayload> {
  return requestJson<RunDetailPayload>(`/api/v1/ops/runs/${encodeURIComponent(runKey)}`, "run detail request failed");
}

export interface SignalsPayload {
  meta: import("./types").MetaPayload;
  signals: {
    activation_peaks: Record<string, number>;
    latency_trackers: Record<string, {
      active_count: number;
      active_components: string[];
      slots: Array<{
        task_id: string;
        component: string;
        elapsed_sec: number;
        co_located_at_entry: string[];
        max_concurrent: number;
        currently_solo: boolean;
        solo_time_acc_sec: number;
      }>;
    }>;
    interference: {
      solo_baselines: Record<string, { median_sec: number; n: number }>;
      solo_vram: Record<string, { median_mib: number; n: number }>;
      pairwise: Record<string, {
        slowdown_median: number | null;
        vram_overhead_median: number | null;
        gpu_util_share_median: number | null;
        n_slowdown: number;
        n_vram: number;
        n_gpu_util: number;
        confidence: number;
      }>;
      self_slowdown?: Record<string, {
        n_obs: number;
        n_valid: number;
        median_delta: number;
        mean_delta: number;
        recent_samples: Array<{
          input_size: number;
          n_concurrency: number;
          delta: number;
          gpu_id: string;
          observed_at: number;
        }>;
      }>;
      workload_classes: Record<string, string>;
      config: Record<string, unknown>;
    };
    workload_profiles: Record<string, {
      workload_class: string;
      mean_gpu_util_percent: number | null;
      mean_execute_us: number | null;
      mean_active_memory_mib: number | null;
      sample_count: number;
    }>;
    resource_profiles: Record<string, {
      component: string;
      workload_class: string;
      gpu_util_ema: number | null;
      arithmetic_intensity: number | null;
      sample_count: number;
      vram_confidence: number;
      latency_confidence: number;
      config_baselines: Record<string, {
        config_fingerprint: string;
        gpu_baselines: Record<string, {
          config_fingerprint: string;
          vram_mib: {
            mean: number; std: number; n: number; confidence: number;
            type?: string; kernel?: string; lengthscale?: number; sigma2_f?: number;
            beta0?: number; beta1?: number; n_total?: number;
            observations?: { x: number[]; y: number[] };
            posterior_curve?: { x: number[]; mean: number[]; upper_2sigma: number[]; lower_2sigma: number[] } | null;
          };
          latency_sec: {
            mean: number; std: number; n: number; confidence: number;
            type?: string; kernel?: string; lengthscale?: number; sigma2_f?: number;
            beta0?: number; beta1?: number; n_total?: number;
            observations?: { x: number[]; y: number[] };
            posterior_curve?: { x: number[]; mean: number[]; upper_2sigma: number[]; lower_2sigma: number[] } | null;
          };
          campaigns_observed: number;
          total_observations: number;
        }>;
      }>;
    }>;
    campaign_scheduler?: {
      primary_campaign: string | null;
      campaign_count: number;
      campaigns: Record<string, {
        campaign_id: string;
        arrival_time: number;
        pending_tasks: number;
        active_tasks: number;
        completed_tasks: number;
        succeeded_tasks?: number;
        failed_tasks?: number;
        cancelled_tasks?: number;
        remaining_est_sec?: number | null;
        is_dag_complete?: boolean;
        observed_fan_outs: Record<string, number[]>;
        fan_out_assignments: Record<string, string[]>;
      }>;
      gpu_timelines: Record<string, {
        gpu_id: string;
        total_vram_mb: number;
        current_reserved_mb: number;
        current_available_mb: number;
        entry_count: number;
        entries: Array<{
          task_id: string;
          component: string;
          campaign_id: string;
          gpu_id: string;
          start_time: number;
          predicted_end_time: number;
          predicted_vram_mb: number;
          is_backfill: boolean;
          is_predicted: boolean;
          is_init: boolean;
          prediction_stale: boolean;
          completed: boolean;
          completed_at: number | null;
          is_killed: boolean;
          killed_at: number | null;
          elapsed_sec: number;
          remaining_sec: number;
        }>;
      }>;
    } | null;
    scheduling_scenario?: {
      snapshot_at: number;
      gpu_views: Record<string, {
        gpu_id: string;
        total_vram_mib: number;
        reserved_vram_mib: number;
        available_vram_mib: number;
        utilization_pct: number;
        inflight_components: string[];
        projected_clear_sec: number;
      }>;
      component_envelopes: Record<string, {
        component: string;
        vram_p50_mib: number;
        vram_p95_mib: number;
        latency_p50_sec: number;
        latency_p95_sec: number;
        workload_class: string;
        vram_confidence: number;
        latency_confidence: number;
        vram_relative_error: number | null;
        latency_relative_error: number | null;
        confidence: number;
        confidence_label: string;
        n_observations: number;
        drift_stale: boolean;
        drift_events_count: number;
      }>;
      colocation_map: Record<string, string[]>;
      n_drift_events_total: number;
      n_stale_envelopes: number;
    } | null;
    init_profiles?: Record<string, Record<string, {
      n: number;
      mu_sec: number;
      sigma_sec: number;
      confidence: number;
    }>>;
  };
}

export async function fetchSignals(): Promise<SignalsPayload> {
  return requestJson<SignalsPayload>("/api/v1/ops/signals", "signals request failed");
}
