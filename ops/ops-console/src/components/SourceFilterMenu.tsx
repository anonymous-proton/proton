import { formatObservedRange } from "../lib/format";
import type { SourceFilterKey } from "../lib/sourceFilters";
import type { CampaignFilters, GatewayAvailableFilters } from "../types";

function ObservedRangeLine({
  firstSeenAt,
  lastSeenAt,
}: {
  firstSeenAt?: string | null;
  lastSeenAt?: string | null;
}) {
  const summary = formatObservedRange(firstSeenAt, lastSeenAt);
  if (summary === "-") {
    return null;
  }
  return (
    <span className="small-muted" title={`${firstSeenAt ?? "-"} -> ${lastSeenAt ?? "-"}`}>
      {summary}
    </span>
  );
}

interface SourceFilterMenuProps {
  sourceDrawerOpen: boolean;
  activeSourceFilterCount: number;
  sourceFilters: CampaignFilters;
  availableGatewayFilters: GatewayAvailableFilters;
  onToggleDrawer: () => void;
  onResetFilters: () => void;
  onToggleSourceFilter: (key: SourceFilterKey, oppositeKey: SourceFilterKey, token: string) => void;
  onSelectAllSourceFilters: (key: SourceFilterKey, oppositeKey: SourceFilterKey, tokens: string[]) => void;
  onClearSourceFilters: (key: SourceFilterKey, oppositeKey: SourceFilterKey) => void;
}

export function SourceFilterMenu({
  sourceDrawerOpen,
  activeSourceFilterCount,
  sourceFilters,
  availableGatewayFilters,
  onToggleDrawer,
  onResetFilters,
  onToggleSourceFilter,
  onSelectAllSourceFilters,
  onClearSourceFilters
}: SourceFilterMenuProps) {
  return (
    <div className="source-menu">
      <button className={`source-menu__button ${sourceDrawerOpen ? "is-open" : ""}`} onClick={onToggleDrawer} type="button">
        <span className="source-menu__icon" aria-hidden="true">
          <span />
          <span />
          <span />
        </span>
        <span>sources</span>
        {activeSourceFilterCount > 0 ? <span className="chip">{activeSourceFilterCount}</span> : null}
      </button>
      {sourceDrawerOpen ? (
        <div className="source-menu__panel">
          <div className="source-menu__panel-header">
            <div>
              <strong>Source Filters</strong>
              <p className="small-muted">Select gateway instances and commits for both queue and runs.</p>
            </div>
            {activeSourceFilterCount > 0 ? (
              <button className="button-secondary" onClick={onResetFilters} type="button">
                clear filters
              </button>
            ) : null}
          </div>
          <div className="source-menu__section">
            <div className="source-menu__section-header">
              <div className="small-muted">gateway</div>
              {availableGatewayFilters.instances.length > 0 ? (
                <div className="source-menu__section-actions">
                  <button
                    className="button-secondary"
                    onClick={() =>
                      onSelectAllSourceFilters(
                        "include_gateway_instance_id",
                        "exclude_gateway_instance_id",
                        availableGatewayFilters.instances.map((option) => option.instance_id)
                      )
                    }
                    type="button"
                  >
                    select all
                  </button>
                  <button
                    className="button-secondary"
                    onClick={() => onClearSourceFilters("include_gateway_instance_id", "exclude_gateway_instance_id")}
                    type="button"
                  >
                    clear all
                  </button>
                </div>
              ) : null}
            </div>
            {availableGatewayFilters.instances.length > 0 ? (
              <div className="source-option-list">
                {availableGatewayFilters.instances.map((option) => (
                  <div key={option.instance_id} className="source-option-row">
                    <div className="source-option-row__label">
                      <strong>{option.label}</strong>
                      <span className="small-muted">count {option.count}</span>
                      <ObservedRangeLine firstSeenAt={option.first_seen_at} lastSeenAt={option.last_seen_at} />
                    </div>
                    <label className="toggle-control source-option-row__toggle">
                      <input
                        type="checkbox"
                        checked={(sourceFilters.include_gateway_instance_id ?? []).includes(option.instance_id)}
                        onChange={() =>
                          onToggleSourceFilter("include_gateway_instance_id", "exclude_gateway_instance_id", option.instance_id)
                        }
                      />
                      <span>include</span>
                    </label>
                  </div>
                ))}
              </div>
            ) : (
              <p className="small-muted">No gateway instances observed yet.</p>
            )}
          </div>
          <div className="source-menu__section">
            <div className="source-menu__section-header">
              <div className="small-muted">commit</div>
              {availableGatewayFilters.commits.length > 0 ? (
                <div className="source-menu__section-actions">
                  <button
                    className="button-secondary"
                    onClick={() =>
                      onSelectAllSourceFilters(
                        "include_gateway_git_commit",
                        "exclude_gateway_git_commit",
                        availableGatewayFilters.commits.map((option) => option.git_commit)
                      )
                    }
                    type="button"
                  >
                    select all
                  </button>
                  <button
                    className="button-secondary"
                    onClick={() => onClearSourceFilters("include_gateway_git_commit", "exclude_gateway_git_commit")}
                    type="button"
                  >
                    clear all
                  </button>
                </div>
              ) : null}
            </div>
            {availableGatewayFilters.commits.length > 0 ? (
              <div className="source-option-list">
                {availableGatewayFilters.commits.map((option) => (
                  <div key={option.git_commit} className="source-option-row">
                    <div className="source-option-row__label">
                      <strong>{option.label}</strong>
                      <span className="small-muted">count {option.count}</span>
                      <ObservedRangeLine firstSeenAt={option.first_seen_at} lastSeenAt={option.last_seen_at} />
                    </div>
                    <label className="toggle-control source-option-row__toggle">
                      <input
                        type="checkbox"
                        checked={(sourceFilters.include_gateway_git_commit ?? []).includes(option.git_commit)}
                        onChange={() => onToggleSourceFilter("include_gateway_git_commit", "exclude_gateway_git_commit", option.git_commit)}
                      />
                      <span>include</span>
                    </label>
                  </div>
                ))}
              </div>
            ) : (
              <p className="small-muted">No commits observed yet.</p>
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}
