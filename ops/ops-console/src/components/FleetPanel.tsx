import { fmt } from "../lib/format";

interface FleetPanelProps {
  queueByComponent: Array<Record<string, unknown>>;
  selectedComponent: string;
  totalQueueDepth: unknown;
  totalInflight: unknown;
  onSelectComponent: (component: string) => void;
}

export function FleetPanel({ queueByComponent, selectedComponent, totalQueueDepth, totalInflight, onSelectComponent }: FleetPanelProps) {
  return (
    <section className="panel panel--compact">
      <div className="section-header">
        <div>
          <h2>Fleet</h2>
          <p className="small-muted">Click a component row to combine fleet filtering with the selected campaign.</p>
        </div>
        <div className="small-muted">
          queue_depth={fmt(totalQueueDepth)} inflight={fmt(totalInflight)}
        </div>
      </div>

      <div className="table-wrap table-wrap--bounded table-wrap--fleet">
        <table className="compact-table">
          <thead>
            <tr>
              <th>component</th>
              <th>workers</th>
              <th>queue</th>
              <th>inflight</th>
            </tr>
          </thead>
          <tbody>
            {queueByComponent.map((row, idx) => {
              const componentName = String(row.component || "");
              const selected = componentName === selectedComponent;
              return (
                <tr key={`${componentName || "component"}-${idx}`} className={selected ? "is-selected" : ""} onClick={() => onSelectComponent(selected ? "" : componentName)}>
                  <td>{fmt(componentName)}</td>
                  <td>{fmt(row.workers)}</td>
                  <td>{fmt(row.queue_depth)}</td>
                  <td>{fmt(row.inflight)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
