import { badgeLabel, badgeTone } from "../lib/status";

export function Badge({ value }: { value: string }) {
  return <span className={`badge badge--${badgeTone(value)}`}>{badgeLabel(value)}</span>;
}
