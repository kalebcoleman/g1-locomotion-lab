# G1 Locomotion Lab: current status

_Last updated 2026-11-10. Simulation only. The repository is public and these results are merged. Unless noted,
every number is one seed and one training run, and the held-out set was not used to choose checkpoints._

## Results

- **Walking.** The final policy keeps **237 of 256** robots up through a 24 s scripted stand, walk, turn and stop
  test with random pushes (reruns 231 and 233); most remaining falls happen while turning
  ([`docs/WALKING.md`](docs/WALKING.md#results-at-a-glance)).
- **Getting up.** The baseline **`recC`** (`recovery/checkpoints/getup_baseline_recC_it1425.pt`) passes full
  criteria v3 on **174.7 of 256 held-out poses (68.2%)**, mean of three evaluations (176 / 174 / 174), up from 129.3
  (50.5%) for the previous baseline. By direction: side A 52, prone 52, side B 36, supine 35 of 64 each. Hard-limit
  steps fell from 1.16% to 0.12%; self-penetration passes on 236 of 236 robots examined
  ([`recovery/checkpoints/README.md`](recovery/checkpoints/README.md)).
- **Hand-over, get-up then walking.** Measured with the **previous** baseline `R9hg_it1100`, the policy the runs
  used: **6 of 250** induced falls complete the sequence under the v3 gate, **28.7%** (pooled over three repeats) with the arms and fingers
  blended to the stance, **99.9%** when the hand-over also waits for a readiness check (a looser check than v3)
  ([M2](pilots/m2_fall_handover_20261108/README.md), [M2b](pilots/m2b_handover_fixes_20261108/README.md)). A rerun
  with `recC` is in progress; no result is recorded yet.
- **Pushes.** Half the robots fall at about **2.2 m/s** of added speed (M50 2.16 m/s, 95% CI 2.08-2.24), about four
  times the largest training push. Left and rear-left pushes are weakest (about 1.4 m/s).
- **Friction.** M50 falls from 2.16 m/s at friction 1.0 to 1.45 m/s at 0.3. Without pushes the policy still walks
  the first 3 s in 98-99% of cases down to friction 0.4, and in 56% at 0.2.
- **Terrain.** Blind walking, no height scan, trained on flat ground: at least 80% survival on **19 degree downhill,
  9 degree uphill, 5.8 cm steps down, 2.8 cm steps up and 8 cm rough ground**
  ([M3](pilots/m3_terrain_baseline_20261108/README.md), [M4](pilots/m4_robustness_20261108/README.md)).

## Open problems

1. **The get-up straddles.** Every standing policy in this lineage pushes its feet about 1.4 m apart, about 0.4 s into
   the rise, before settling to a normal stance. Reward-only changes did not remove it (stance weights to -32,
   smoothness penalties, joint-target bounds), and the current baseline leaves the peak spread unchanged.
2. **Upward steps and steep climbs.** Upward stairs above 2.8 cm and uphill slopes above 9 degrees fall below 80%
   survival; downward terrain is handled much better.
3. **Single seed, single lineage.** Walking and get-up each have one training run. No run-to-run variance or
   confidence interval is claimed for either headline number.
4. **The hand-over has not been re-measured with `recC`.** Its published numbers belong to the previous baseline, and
   its 99.9% figure uses the looser readiness gate rather than the v3 result.

## Next

- Rerun M2/M2b with `RECOVERY_CKPT=recovery/checkpoints/getup_baseline_recC_it1425.pt` and record the result next to
  the previous one.
- Repeat the walking and get-up evaluations on fresh seeds before calling either number stable.
- Test a reference-motion or imitation term for the straddle; reward shaping alone has been ruled out.
- Leave `TEST` untouched until a method is frozen.

Details: [`README.md`](README.md) · [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md) ·
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) · [`pilots/README.md`](pilots/README.md) ·
[`recovery/STATUS.md`](recovery/STATUS.md) (frozen 2026-11-06 get-up snapshot).
