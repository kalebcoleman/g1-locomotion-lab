# Upper-body reference check (2026-11-06, requested by Kaleb). Script: `recovery/scripts/check_links.py`; data: `standing.json`, `sitting.json`, `fold.json` (16 envs each); the five training runs were not touched.
Poses: standing = walking policy, 3 s, standing start; sitting = Lane A v1 policy, 8 s, DEV poses; fold = R6c iteration 350, 5 s, DEV poses.

| reference | standing z (m) | sitting z (m) | fold z (m) | fold/stand contrast | relative to torso_link (spread across envs, all poses) |
|---|---|---|---|---|---|
| pelvis link origin | 0.693 | 0.171 | 0.521 | 0.17 | n/a |
| torso_link, head_link, logo_link **frame origins** | 0.693 | 0.171 | 0.521 | 0.17 | coincide with pelvis origin (USD joint offsets are zero), so NOT an upper-body height |
| imu_link | 0.846 | 0.313 | 0.379 | 0.47 | (-0.0423, 0.0017, 0.1521), spread 0.0000, rotation 0.000 |
| left/right shoulder_pitch_link | 0.95 | 0.41-0.44 | 0.27 | 0.68 | (-0.0025, +-0.104, 0.2593), spread 0.0000 |
| torso_link COM | 0.862 | 0.337 | 0.356 | 0.51 | local (0.002, 0.003, 0.169) |
| **head_link COM** | **1.113** | **0.582** | **0.112** | **1.00** | local (0.0014, 0.0003, 0.4203) |
Torso up-vector z (= pelvis up within 0.02 except with torso_joint offset): 1.00 / 0.98 / -0.97.

**Verdict.** `imu_link` PASSES: rigid on torso_link (zero spread in all poses, zero relative rotation), moves relative to the pelvis with torso_joint (spread 0.014 m in the fold vs 0.0008 standing), articulation data equals Newton's raw `body_q` (env 0: 0.8503 / 0.3045 / 0.3579 both), and its offset matches the USD `imu_joint` (-0.042339, 0.00166, 0.152067). So the R7h reward computes what it claims.
**Limitation flagged:** imu_link is only 0.15 m above the torso frame origin, so it is a low-torso reference. The R7h ramp still pays 0.37 of its maximum in the fold (0.38 m) and 1.0 standing. If R7h stalls, that is not evidence that an upper-body height reward cannot work: **head_link COM (`body_com_pos_w`) is a far stronger reference** (1.11 / 0.58 / 0.11 m; its ramp would pay ~0.01 in the fold). Its local COM comes from the USD mass properties (0.4203 m; standing 0.693 + 0.420 = 1.113 matches the simulation), and the walking/recovery configs do not randomise it (`base_com=None`; `add_base_mass` adds to torso_link only). imu_link itself has mass 1e-6 and an unset COM in the USD, so only its frame is meaningful.
Also: `Episode` code that uses `head_link` link position as a head height (none in this repo's rewards) would silently be a pelvis height.
