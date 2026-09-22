"""
Reservation slot data structures for the Slurm-style baseline.

Port of slurm/src/plugins/sched/backfill/backfill.c::node_space_map_t
@ slurm-23-11-11-1 (commit upstream) — temporal map of GPU-VRAM
availability used to drive the EASY-backfill window check.

Upstream `node_space_map_t` (backfill.c:132-138) is a per-time-slice
record:
  begin_time   — slice start
  end_time     — slice end
  avail_bitmap — bitmask of nodes available in this slice
  next         — index of the next slice (zero-terminated chain)

We collapse that to a per-(gpu_id, time-window, vram-amount, job_id)
record because our resource is GPU memory rather than node count, and
the gateway's worker pool is fixed-size (no node admit/withdraw).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReservationSlot:
    """A future GPU-VRAM commitment.

    Mirrors slurm/.../backfill.c::node_space_map_t @ slurm-23-11-11-1
    in that each slot carries a [begin_time, end_time] window and the
    resource amount committed in that window.

    Slurm's avail_bitmap is bit-OR'd across the cluster's nodes; in
    our environment "node" = GPU, so each ReservationSlot is per-GPU.
    """

    gpu_id: str
    start_time: float
    end_time: float
    vram_mb: int
    job_id: str

    def overlaps(self, other_start: float, other_end: float) -> bool:
        """Time-window overlap test.

        Mirrors slurm/.../backfill.c::_test_resv_overlap @ 3496-3526:
            (node_space[j].end_time > start_time) &&
            (node_space[j].begin_time < end_reserve)
        """
        return (self.end_time > other_start) and (self.start_time < other_end)
