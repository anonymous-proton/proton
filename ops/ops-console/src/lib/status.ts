export function isTerminalCampaignStatus(status: string): boolean {
  const normalized = status.trim().toLowerCase();
  return normalized === "completed" || normalized === "failed";
}

export function badgeTone(value: string): string {
  const normalized = value.toLowerCase();
  if (
    normalized === "predicted_ok" ||
    normalized === "ok" ||
    normalized === "succeeded" ||
    normalized === "exact" ||
    normalized === "post_selection"
  ) {
    return "success";
  }
  if (
    normalized === "predicted_violated" ||
    normalized === "missing_worker_context" ||
    normalized === "failed" ||
    normalized === "cancelled"
  ) {
    return "danger";
  }
  if (
    normalized === "abstained" ||
    normalized === "telemetry_missing" ||
    normalized === "post_selection_retry" ||
    normalized === "nearby" ||
    normalized === "active" ||
    normalized === "running" ||
    normalized.startsWith("no_")
  ) {
    return "warning";
  }
  if (normalized === "coarse" || normalized === "none" || normalized === "unknown") {
    return "neutral";
  }
  return "info";
}

export function badgeLabel(value: string): string {
  const normalized = value.trim().toLowerCase();
  if (!normalized) {
    return value;
  }
  const labels: Record<string, string> = {
    predicted_ok: "predicted ok",
    predicted_violated: "guard miss",
    telemetry_missing: "telemetry missing",
    post_selection: "selected worker",
    post_selection_retry: "reselected worker",
    missing_worker_context: "missing worker context",
    no_qc_kept_rows: "no QC-kept rows",
    no_history_rows: "no history rows",
    no_compatible_rows: "no compatible rows"
  };
  return labels[normalized] ?? normalized.replace(/_/g, " ");
}
