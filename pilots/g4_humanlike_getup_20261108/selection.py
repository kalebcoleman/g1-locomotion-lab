"""Pure selection rule used by driver_g4.py (CPU-testable). Rows are dicts with 'v3' (DEV criteria-v3 count) and 'peak_spread' (mean peak foot spread, m)."""

def pick_stage_best(rows, need):
    """A checkpoint is FEASIBLE if v3 >= need (a fraction of the baseline). The stage-best is the feasible checkpoint with the smallest peak spread, else (none feasible) the one with the
    highest v3. Returns (pick or None, list of feasible rows)."""
    feas = [r for r in rows if r['v3'] >= need]
    if feas: return min(feas, key=lambda r: r['peak_spread']), feas
    return (max(rows, key=lambda r: r['v3']) if rows else None), feas


def target_met(pick, feas, spread_target):
    """The spread target counts only for a feasible checkpoint: a policy that never stands has a small spread trivially (the bug that ended the first discovery runs early)."""
    return bool(feas) and pick is not None and pick['peak_spread'] < spread_target
