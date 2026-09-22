import type { GatewaySourceSummary, SupportDistribution } from "../types";

const RELATIVE_TIME_FORMATTER = new Intl.RelativeTimeFormat("en", { numeric: "auto" });

export function asRecord(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  return value as Record<string, unknown>;
}

export function fmt(value: unknown, digits = 3): string {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  if (typeof value === "number") {
    return Number.isFinite(value) ? value.toFixed(digits) : "-";
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  return String(value);
}

export function asNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

export function fmtPercent(value: unknown): string {
  const numeric = asNumber(value);
  if (numeric === null) {
    return "-";
  }
  return `${(numeric * 100).toFixed(0)}%`;
}

export function displayUnit(unit: string): string {
  return unit === "mib" ? "MiB" : unit;
}

export function fmtWithUnit(value: number | null | undefined, unit: string, digits = 1): string {
  const numeric = asNumber(value);
  if (numeric === null) {
    return "-";
  }
  return `${numeric.toFixed(digits)} ${displayUnit(unit)}`;
}

export function fmtSignedWithUnit(value: number | null | undefined, unit: string, digits = 1): string {
  const numeric = asNumber(value);
  if (numeric === null) {
    return "-";
  }
  const prefix = numeric > 0 ? "+" : "";
  return `${prefix}${numeric.toFixed(digits)} ${displayUnit(unit)}`;
}

export function formatDistribution(distribution: SupportDistribution): string {
  return `${distribution.exact}/${distribution.nearby}/${distribution.coarse}/${distribution.none}`;
}

export function formatRelativeTime(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) {
    return "-";
  }
  const diffSeconds = Math.round((timestamp - Date.now()) / 1000);
  const absSeconds = Math.abs(diffSeconds);
  if (absSeconds < 5) {
    return "just now";
  }
  if (absSeconds < 60) {
    return RELATIVE_TIME_FORMATTER.format(diffSeconds, "second");
  }
  const diffMinutes = Math.round(diffSeconds / 60);
  if (absSeconds < 3600) {
    return RELATIVE_TIME_FORMATTER.format(diffMinutes, "minute");
  }
  const diffHours = Math.round(diffSeconds / 3600);
  if (absSeconds < 86400) {
    return RELATIVE_TIME_FORMATTER.format(diffHours, "hour");
  }
  const diffDays = Math.round(diffSeconds / 86400);
  if (absSeconds < 604800) {
    return RELATIVE_TIME_FORMATTER.format(diffDays, "day");
  }
  const diffWeeks = Math.round(diffSeconds / 604800);
  if (absSeconds < 2592000) {
    return RELATIVE_TIME_FORMATTER.format(diffWeeks, "week");
  }
  const diffMonths = Math.round(diffSeconds / 2592000);
  if (absSeconds < 31536000) {
    return RELATIVE_TIME_FORMATTER.format(diffMonths, "month");
  }
  return RELATIVE_TIME_FORMATTER.format(Math.round(diffSeconds / 31536000), "year");
}

export function formatObservedRange(firstSeenAt: string | null | undefined, lastSeenAt: string | null | undefined): string {
  const parts: string[] = [];
  const oldest = formatRelativeTime(firstSeenAt);
  const latest = formatRelativeTime(lastSeenAt);
  if (oldest !== "-") {
    parts.push(`oldest ${oldest}`);
  }
  if (latest !== "-") {
    parts.push(`latest ${latest}`);
  }
  return parts.join(" | ") || "-";
}

export function sourceSummaryLine(summary: GatewaySourceSummary[]): string {
  if (summary.length === 0) {
    return "-";
  }
  const first = summary[0];
  const firstLabel = String(first.label || "unknown");
  if (summary.length === 1) {
    return firstLabel;
  }
  return `${firstLabel} +${summary.length - 1}`;
}
