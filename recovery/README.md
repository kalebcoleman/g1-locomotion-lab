# Fall recovery: videos, pictures and architecture

Work in progress. Simulation only: Isaac Lab 3.0.0 with Newton / MuJoCo-Warp, Unitree G1 (full collision, self-collision on, Unitree's real torque limits). Every clip is an uncut 10 s episode from a fixed evaluation pose with the training assist off. Contact sheets show t = 0.5, 1, 2, 4, 6 and 9 s.

**Status (2026-11-07):** the best policy (R9hg, iteration 1100) passes the full validity criteria v3 on 127-129 of 256 held-out poses (49.6-50.4%) across three evaluations. That is one seed, produced by a warm-start chain; prone about 78%, side A about 50%, supine about 46%, side B about 27%. The reserved TEST set is untouched, and the hand-over to the walking policy has not been tested yet. Replication on two more seeds and a port of HoST are training.

**Criteria v3** (all required): pelvis at least 0.60 m with the torso within 26 degrees of vertical and both feet carrying load, held for 2 s; no launch (upward pelvis speed at most 2 m/s); at most 2% of steps without ground contact; joint speed limits broken on at most 2% of steps; soft joint limits on at most 10% and hard limits on at most 3% of steps (at most 0.10 rad past); torque within Unitree's limits on at least 97% of steps; no limb passing through the body; feet at most 0.60 m apart; still standing in the final second; no assist force.

<p align="center"><img src="media/recovery_best_close.gif" width="560" alt="Best policy so far, four lying starts"></p>

<p align="center"><img src="media/recovery_before_after.gif" alt="Earlier forward fold (left) and current best (right), same 16 starts"></p>

## Architecture

![Recovery system](architecture/figures/recovery_system_dark.png)

![Networks](architecture/figures/network_architecture_dark.png)

Plain-language overview, PPO settings and version names: [architecture/SYSTEM_OVERVIEW.md](architecture/SYSTEM_OVERVIEW.md). Editable diagrams: [system](architecture/recovery_system_diagram.html), [networks](architecture/network_architecture_diagram.html).

## Plots

![Training curves at the v15 check](media/20261106_235_chkv15_curves.png)

![Postures reached at the v15 check](media/20261106_235_chkv15_postures.png)

![Anatomy of the rejected R7m iteration-350 stands](media/20261106_235_chkv15_R7m_it350_anatomy.png)

## Current best: R9hg, iteration 1100

| contact sheet | video | what it shows | verdict |
|---|---|---|---|
| <img src="media/20261107_03_R9hg_it1100_close_sheet.png" width="360"> | [20261107_03_R9hg_it1100_close.mp4](media/20261107_03_R9hg_it1100_close.mp4) | 4 DEV poses, close-up. Rises in about 1 s, upright, normal stance, only the feet touch the floor, holds to 10 s. | **Best so far:** full v3 on 127-129/256 HELDOUT poses (49.6-50.4%), three evaluations, one seed |
| <img src="media/20261107_03_R9hg_it1100_dev16_sheet.png" width="360"> | [20261107_03_R9hg_it1100_dev16.mp4](media/20261107_03_R9hg_it1100_dev16.mp4) | First 16 DEV poses, all four fall directions. | Same checkpoint; side B is the weakest direction (about 27%) |

## First robots to pass every v3 rule

| contact sheet | video | what it shows | verdict |
|---|---|---|---|
| <img src="media/20261107_02_R8wW_it850_v3poses_sheet.png" width="360"> | [20261107_02_R8wW_it850_v3poses.mp4](media/20261107_02_R8wW_it850_v3poses.mp4) | DEV poses 5 (prone) and 16 (side), plus two others, R8w warm iteration 850. | Poses 5 and 16 pass all of v3 incl. self-penetration; about 1% of DEV |
| <img src="media/20261107_02_R8wW_it850_close_sheet.png" width="360"> | [20261107_02_R8wW_it850_close.mp4](media/20261107_02_R8wW_it850_close.mp4) | Stance penalty narrows the straddle to about 0.45 m. | 3-7/256 v3 on DEV |
| <img src="media/20261107_02_R8wW_it850_dev16_sheet.png" width="360"> | [20261107_02_R8wW_it850_dev16.mp4](media/20261107_02_R8wW_it850_dev16.mp4) | 16 DEV poses. | as above |

<details>
<summary><b>Failure modes that the criteria caught</b></summary>

| contact sheet | video | what it shows | verdict |
|---|---|---|---|
| <img src="media/20261107_02_R9h_it900_close_sheet.png" width="360"> | [20261107_02_R9h_it900_close.mp4](media/20261107_02_R9h_it900_close.mp4) | Looks like a good stand, but the left hand sits inside the right hip (self-collision was off). | **Rejected:** fails the self-penetration check on every robot checked |
| <img src="media/20261107_02_R9h_it900_dev16_sheet.png" width="360"> | [20261107_02_R9h_it900_dev16.mp4](media/20261107_02_R9h_it900_dev16.mp4) | 16 DEV poses. | Rejected, as above |
| <img src="media/20261107_001_R7m_it500_sheet.png" width="360"> | [20261107_001_R7m_it500.mp4](media/20261107_001_R7m_it500.mp4) | Stable stand from every direction, feet 0.72 m apart. | **Rejected:** ankle roll pressed against its hard stop on 75% of steps |
| <img src="media/20261107_001_R7h_it500_sheet.png" width="360"> | [20261107_001_R7h_it500.mp4](media/20261107_001_R7h_it500.mp4) | Same with feet 0.89 m apart. | **Rejected:** ankle at its stop, torque above Unitree's limits |
| <img src="media/20261106_235_chkv15_R7m_it350_close_sheet.png" width="360"> | [20261106_235_chkv15_R7m_it350_close.mp4](media/20261106_235_chkv15_R7m_it350_close.mp4) | Fast leg splay into a wide straddle, then drops to a kneel at about 8 s. | **Rejected:** passed the older v2.1 on 21-29 poses but not a stable stand; torque 3-6x real limits |
| <img src="media/20261106_235_chkv15_R7m_it350_sheet.png" width="360"> | [20261106_235_chkv15_R7m_it350.mp4](media/20261106_235_chkv15_R7m_it350.mp4) | 16 DEV poses. | Rejected, as above |

</details>

<details>
<summary><b>Reward-design progression (iteration-300 checks)</b></summary>

| contact sheet | video | what it shows | verdict |
|---|---|---|---|
| <img src="media/20261106_235_chkv15_R7m_it300_sheet.png" width="360"> | [20261106_235_chkv15_R7m_it300.mp4](media/20261106_235_chkv15_R7m_it300.mp4) | Height reward gated by torso orientation: first robots crouch on their feet. | No stand |
| <img src="media/20261106_235_chkv15_R7h_it300_sheet.png" width="360"> | [20261106_235_chkv15_R7h_it300.mp4](media/20261106_235_chkv15_R7h_it300.mp4) | Height measured at the IMU instead of the pelvis. | No stand |
| <img src="media/20261106_235_chkv15_R7u_it300_sheet.png" width="360"> | [20261106_235_chkv15_R7u_it300.mp4](media/20261106_235_chkv15_R7u_it300.mp4) | Orientation reward with a gradient below horizontal. | Forward fold |
| <img src="media/20261106_235_chkv15_R7uW_it650_sheet.png" width="360"> | [20261106_235_chkv15_R7uW_it650.mp4](media/20261106_235_chkv15_R7uW_it650.mp4) | Same, warm-started from the fold. | Forward fold |
| <img src="media/20261106_235_chkv15_R6c_it750_sheet.png" width="360"> | [20261106_235_chkv15_R6c_it750.mp4](media/20261106_235_chkv15_R6c_it750.mp4) | Height-dominant reward, lying-only starts (control). | Forward fold: pelvis high, torso upside down |
| <img src="media/20261106_224419_chk300_R6cS_t1_3_6_9.png" width="360"> | [20261106_224419_chk300_R6cS.mp4](media/20261106_224419_chk300_R6cS.mp4) | Earlier control. | Forward fold |
| <img src="media/20261106_224419_chk300_R6b_t1_3_6_9.png" width="360"> | [20261106_224419_chk300_R6b.mp4](media/20261106_224419_chk300_R6b.mp4) | Height 10, upright 2. | Fold |
| <img src="media/20261106_224419_chk300_R5_t1_3_6_9.png" width="360"> | [20261106_224419_chk300_R5.mp4](media/20261106_224419_chk300_R5.mp4) | Larger action scale and noise. | Sits |
| <img src="media/20261106_224419_chk300_R4n_t1_3_6_9.png" width="360"> | [20261106_224419_chk300_R4n.mp4](media/20261106_224419_chk300_R4n.mp4) | Lying-dominant starts, no assist. | Sits |
| <img src="media/20261106_224419_chk300_R4nS_t1_3_6_9.png" width="360"> | [20261106_224419_chk300_R4nS.mp4](media/20261106_224419_chk300_R4nS.mp4) | Same from scratch. | Sits |

</details>

<details>
<summary><b>Earlier rounds (before the current criteria)</b></summary>

| image | what it shows |
|---|---|
| <img src="media/earlier/20261106_165529_heldout_poses_t0.3s.png" width="360"> | the held-out lying poses at 0.3 s |
| <img src="media/earlier/20261106_171247_A_v1_failure.gif" width="360"> | Lane A v1: sits up, never stands |
| <img src="media/earlier/20261106_171247_A_v1_failure_t9s.png" width="360"> | Lane A v1: sits up, never stands |
| <img src="media/earlier/20261106_173906_level1_start_poses.png" width="360"> | sit-bank start poses used by the curriculum |
| <img src="media/earlier/20261106_183606_A6_heldout_t3.5s.png" width="360"> | A6: rises by launching itself (rejected) |
| <img src="media/earlier/20261106_190102_A8c_heldout_16poses.gif" width="360"> | A8c: the launch exploit that scored 256/256 under the first criteria (rejected: airborne at 0.3 s) |
| <img src="media/earlier/20261106_190102_A8c_heldout_t0.3s.png" width="360"> | A8c: the launch exploit that scored 256/256 under the first criteria (rejected: airborne at 0.3 s) |
| <img src="media/earlier/20261106_190102_A8c_heldout_t1.5s.png" width="360"> | A8c: the launch exploit that scored 256/256 under the first criteria (rejected: airborne at 0.3 s) |
| <img src="media/earlier/20261106_205009_V11_unassisted_16poses.gif" width="360"> | V11 without assist: crouched hand-and-foot postures, no flight |
| <img src="media/earlier/20261106_205009_V11_unassisted_t6s.png" width="360"> | V11 without assist: crouched hand-and-foot postures, no flight |

</details>

