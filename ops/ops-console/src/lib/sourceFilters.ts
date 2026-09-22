import type { CampaignFilters } from "../types";

export type SourceFilterKey =
  | "include_gateway_instance_id"
  | "exclude_gateway_instance_id"
  | "include_gateway_git_commit"
  | "exclude_gateway_git_commit";

export function createEmptySourceFilters(): CampaignFilters {
  return {
    include_gateway_instance_id: [],
    exclude_gateway_instance_id: [],
    include_gateway_git_commit: [],
    exclude_gateway_git_commit: []
  };
}

export function toggleToken(values: string[], token: string): string[] {
  return values.includes(token) ? values.filter((item) => item !== token) : [...values, token];
}

export function sourceFilterCount(filters: CampaignFilters): number {
  return (
    (filters.include_gateway_instance_id?.length ?? 0) +
    (filters.exclude_gateway_instance_id?.length ?? 0) +
    (filters.include_gateway_git_commit?.length ?? 0) +
    (filters.exclude_gateway_git_commit?.length ?? 0)
  );
}
