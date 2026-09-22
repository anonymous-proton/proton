export interface SourceStatus {
  required: boolean;
  status: string;
  stale: boolean;
  error_reason?: string | null;
  last_attempt_at?: string | null;
  last_ok_at?: string | null;
}

export interface MetaPayload {
  generated_at: string;
  data_age_seconds: number;
  stale: boolean;
  stale_reason: string[];
  poll_interval_seconds: number;
  ttl_seconds: number;
  sources: Record<string, SourceStatus>;
}

export interface CampaignTaskCounts {
  submitted: number;
  running: number;
  succeeded: number;
  failed: number;
  cancelled: number;
  observed: number;
  total: number;
}

export interface CampaignSummaryRow {
  campaign_id: string;
  run_name: string;
  submitter: string;
  status: "waiting" | "active" | "completed" | "failed" | "unknown";
  started_at?: string | null;
  last_seen_at?: string | null;
  ended_at?: string | null;
  task_counts: CampaignTaskCounts;
  component_summary: Array<Record<string, unknown>>;
  telemetry_readiness: Record<string, unknown>;
  gateway_summary?: GatewaySourceSummary[];
}

export interface UnassignedSummary {
  rows: number;
  last_seen_at?: string | null;
  reasons: Array<{ reason: string; count: number }>;
}

export interface CampaignQueueSection {
  summary: Record<string, number>;
  count: number;
  campaigns: CampaignSummaryRow[];
  unassigned: UnassignedSummary;
  error_reason?: string | null;
}

export interface QueueSection {
  total_queue_depth: number;
  total_inflight: number;
  by_component: Array<{
    component: string;
    workers: number;
    queue_depth: number;
    inflight: number;
  }>;
}

export interface FleetSection {
  queue: QueueSection;
}

export interface OverviewPayload {
  meta: MetaPayload;
  fleet: FleetSection;
}

export interface GatewaySourceSummary {
  instance_id: string;
  bind_addr: string;
  git_commit: string;
  label: string;
  count: number;
  last_seen_at?: string | null;
}

export interface GatewayFacetInstance {
  instance_id: string;
  label: string;
  bind_addr: string;
  git_commit: string;
  count: number;
  first_seen_at?: string | null;
  last_seen_at?: string | null;
}

export interface GatewayFacetCommit {
  git_commit: string;
  label: string;
  count: number;
  first_seen_at?: string | null;
  last_seen_at?: string | null;
}

export interface GatewayAvailableFilters {
  instances: GatewayFacetInstance[];
  commits: GatewayFacetCommit[];
}

export interface CampaignDetailPayload {
  campaign_id: string;
  run_name: string;
  submitter: string;
  status: string;
  started_at?: string | null;
  last_seen_at?: string | null;
  ended_at?: string | null;
  task_counts?: CampaignTaskCounts;
  overview?: Record<string, unknown>;
  components?: Array<Record<string, unknown>>;
  tasks?: Array<Record<string, unknown>>;
  telemetry_readiness?: Record<string, unknown>;
  gateway_summary?: GatewaySourceSummary[];
}

export interface RecentRunFilters {
  limit?: number;
  campaign_id?: string;
  component?: string;
  worker_name?: string;
  include_gateway_instance_id?: string[];
  exclude_gateway_instance_id?: string[];
  include_gateway_git_commit?: string[];
  exclude_gateway_git_commit?: string[];
}

export interface CampaignFilters {
  include_gateway_instance_id?: string[];
  exclude_gateway_instance_id?: string[];
  include_gateway_git_commit?: string[];
  exclude_gateway_git_commit?: string[];
}

export interface CampaignsPayload extends CampaignQueueSection {
  available_gateway_filters: GatewayAvailableFilters;
}

export interface SupportDistribution {
  exact: number;
  nearby: number;
  coarse: number;
  none: number;
}

export interface EstimatorRunSummary {
  last_1h_runs: number;
  runtime_support_distribution: SupportDistribution;
  active_memory_support_distribution: SupportDistribution;
  resident_baseline_support_distribution: SupportDistribution;
  runtime_violation_count: number;
  active_memory_violation_count: number;
  runtime_abstention_count: number;
  active_memory_abstention_count: number;
  resident_baseline_abstention_count: number;
  campaign_correction_applied_count: number;
  telemetry_missing_count: number;
  post_selection_retry_count: number;
}

export type TargetStatus = "predicted_ok" | "predicted_violated" | "abstained" | "telemetry_missing";

export interface TargetExplanationSummary {
  title: string;
  status: TargetStatus;
  summary: string;
  unit: string;
  scope?: string | null;
  center: number | null;
  upper: number | null;
  actual: number | null;
  guard_margin: number | null;
  delta_vs_upper: number | null;
  support_level: string;
  fallback_level: string;
  effective_support: number;
  abstain_reason_code?: string | null;
  abstain_reason?: string | null;
}

export interface TargetEvidence {
  n_history_rows: number;
  n_exact_rows: number;
  n_nearby_rows: number;
  n_coarse_rows: number;
  n_selected_rows: number;
  n_campaign_correction_rows: number;
}

export interface GuardComponents {
  base: number | null;
  support: number | null;
  fallback: number | null;
  safety: number | null;
  total: number | null;
}

export interface TargetExplanationDetail extends TargetExplanationSummary {
  target_id: string;
  evidence: TargetEvidence;
  guard_components: GuardComponents;
  campaign_correction_total: number | null;
  reason_codes: string[];
}

export interface RecentRunRow {
  time: string | null;
  run_key: string;
  campaign_id: string;
  stage_id: string;
  model_id: string;
  worker_id: string;
  status: string;
  gateway: {
    instance_id: string;
    bind_addr: string;
    git_commit: string;
    started_at?: number | null;
    label: string;
  };
  gateway_label: string;
  selected_context_applied: string;
  selected_worker_context_summary: string;
  local_regime_bucket: string;
  co_location_signature: string;
  takeaway: string;
  runtime: TargetExplanationSummary;
  active_memory: TargetExplanationSummary;
  resident_baseline: TargetExplanationSummary;
  telemetry_missing: boolean;
}

export interface RecentRunsPayload {
  generated_at: string;
  filters: {
    campaign_id?: string | null;
    component?: string | null;
    worker_name?: string | null;
    include_gateway_instance_id?: string[];
    exclude_gateway_instance_id?: string[];
    include_gateway_git_commit?: string[];
    exclude_gateway_git_commit?: string[];
  };
  summary: EstimatorRunSummary;
  total: number;
  limit: number;
  offset: number;
  runs: RecentRunRow[];
}

export interface RequestTraceContext {
  campaign_id: string;
  stage_id: string;
  model_id: string;
  stage_class: string;
  descriptor_bucket: string;
  requested_batch_size?: number | null;
  effective_batch_bucket: string;
}

export interface SelectedWorkerContextDetail {
  worker_id: string;
  hardware_software: string;
  active_request_count_bucket: string;
  co_location_signature: string;
  residency_state: string;
}

export interface RunTakeaway {
  headline: string;
  status: string;
  notes: string[];
}

export interface ActualTelemetrySection {
  actual_hot_execution_sec: number | null;
  actual_transition_penalty_sec: number | null;
  actual_total_runtime_sec: number | null;
  actual_active_memory_mib: number | null;
  dispatch_time_resident_baseline_mib: number | null;
  dispatch_time_resident_baseline_state: string;
  failure_outcome: string;
  runtime_violation: boolean;
  memory_violation: boolean;
  telemetry_missing: boolean;
}

export interface EstimatorExplanationSection {
  runtime: TargetExplanationDetail;
  active_memory: TargetExplanationDetail;
  resident_baseline: TargetExplanationDetail;
  hot_execution_time: TargetExplanationDetail;
  transition_penalty: TargetExplanationDetail;
}

export interface RunDetailPayload {
  generated_at: string;
  run_key: string;
  overview: RecentRunRow;
  takeaway: RunTakeaway;
  request_trace_context: RequestTraceContext;
  selected_worker_context: SelectedWorkerContextDetail;
  estimator_explanations: EstimatorExplanationSection;
  actual_telemetry: ActualTelemetrySection;
  raw: {
    run_record: Record<string, unknown>;
    scheduler_decision: Record<string, unknown>;
    axes: Record<string, unknown>;
  };
}
