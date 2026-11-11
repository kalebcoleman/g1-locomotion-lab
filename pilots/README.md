# Pilot index

Dated, bounded experiments. Each directory keeps its own README, scripts, ledger and result JSONs; nothing here is
trained by the main project, and the get-up baseline lives in [`recovery/checkpoints/`](../recovery/checkpoints/README.md).
The current overall status is in [`STATUS.md`](../STATUS.md).

| Pilot | Question asked | Result in one line | Details |
|---|---|---|---|
| [`humanlike_20261108`](humanlike_20261108/ROADMAP.md) | Can reward shaping alone make the get-up look more human, in particular remove the early-rise foot spread? | **Negative:** the spread stayed at about 1.3 m and success fell (128 -> 64 -> 83 of 256); the fix likely needs reference motion | [roadmap](humanlike_20261108/ROADMAP.md), [hand-off notes](humanlike_20261108/HANDOFF.md) |
| [`m2_fall_handover_20261108`](m2_fall_handover_20261108/README.md) | Can frozen walking and get-up policies chain through an induced fall, a stand gate and back into walking, with resets never credited? | **6 of 250** induced falls complete the sequence under the v3 gate (2.4%); most stop at the gate or re-fall | [README](m2_fall_handover_20261108/README.md) |
| [`m2b_handover_fixes_20261108`](m2b_handover_fixes_20261108/README.md) | Which hand-over fix (settle, posture blend, command ramp, readiness trigger) removes the re-falls? | Blending arms and fingers removes them: **28.7%** full success under the v3 gate (pooled over three repeats), **99.9%** with the readiness trigger (a looser gate) | [README](m2b_handover_fixes_20261108/README.md) |
| [`m2c_handover_recC_20261110`](m2c_handover_recC_20261110/README.md) | Does the better `recC` get-up translate into better end-to-end success? | Yes with the blend: **51.6%** (387 of 750) vs 28.7%; without the blend it drops to 0.4% because the extra hand-overs re-fall | [README](m2c_handover_recC_20261110/README.md) |
| [`m3_terrain_baseline_20261108`](m3_terrain_baseline_20261108/README.md) | How does the flat-trained walking policy handle graded terrain when it cannot see it (no height scan)? | At least **80% survival** on 19 degree downhill, 9 degree uphill, 5.8 cm steps down, 2.8 cm steps up and 8 cm rough ground | [README](m3_terrain_baseline_20261108/README.md) |
| [`m4_robustness_20261108`](m4_robustness_20261108/README.md) | How much push and how little friction does the walking policy survive? | Push **M50 = 2.16 m/s** at friction 1.0, falling to **1.45 m/s at friction 0.3**; left and rear-left pushes are weakest | [README](m4_robustness_20261108/README.md) |
| [`g4_humanlike_getup_20261108`](g4_humanlike_getup_20261108/README.md) | Can a stance penalty and bounded joints remove the straddle while keeping the stand rate, and does continued training stay stable? | **Straddle target not met**, but the stable recipe raised HELDOUT v3 from 129.3 to **174.7 of 256 (68.2%)**; `recC` is the current baseline | [README](g4_humanlike_getup_20261108/README.md) |

M2 and M2b were measured with the previous get-up baseline `R9hg_it1100`; M2c repeats M2b with `recC`. The M2 and M2b
scripts accept `RECOVERY_CKPT=<path>` to rerun them with another checkpoint.
