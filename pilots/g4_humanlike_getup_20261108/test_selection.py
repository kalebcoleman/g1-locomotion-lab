"""CPU tests for the Goal 4 selection rule. Run: python3 test_selection.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selection import pick_stage_best, target_met
R = lambda it, v3, pk: dict(it=it, v3=v3, peak_spread=pk)


def test_feasible_checkpoint_with_the_smallest_spread_wins():
    rows = [R(1, 120, 1.40), R(2, 95, 0.90), R(3, 70, 0.60), R(4, 100, 1.10)]
    pick, feas = pick_stage_best(rows, need=90)
    assert pick['it'] == 2 and [r['it'] for r in feas] == [1, 2, 4]          # it 3 has the smallest spread but is below the v3 bar


def test_without_a_feasible_checkpoint_the_highest_v3_is_picked():
    pick, feas = pick_stage_best([R(1, 40, 0.5), R(2, 60, 1.4), R(3, 20, 0.4)], need=90)
    assert pick['it'] == 2 and feas == []


def test_the_bar_is_inclusive_and_empty_input_is_safe():
    assert pick_stage_best([R(1, 90, 1.0)], need=90)[0]['it'] == 1
    assert pick_stage_best([], need=90) == (None, [])


def test_target_needs_a_feasible_checkpoint():
    pick, feas = pick_stage_best([R(1, 0, 0.44)], need=90)                      # never stands: tiny spread, v3 0
    assert not target_met(pick, feas, 0.8), "a non-standing policy must not meet the spread target"
    pick, feas = pick_stage_best([R(1, 100, 0.75)], need=90); assert target_met(pick, feas, 0.8)
    pick, feas = pick_stage_best([R(1, 100, 0.80)], need=90); assert not target_met(pick, feas, 0.8)       # strictly below
    assert not target_met(None, [], 0.8)


if __name__ == "__main__":
    n = 0
    for k, v in list(globals().items()):
        if k.startswith("test_"): v(); print("ok", k); n += 1
    print(n, "passed")
