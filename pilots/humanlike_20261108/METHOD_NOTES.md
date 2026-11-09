# Human-like floor-to-stand pilot (SIMULATION ONLY), 2026-11-08
Coordinator: G1 bot (Grok Bot). Authorised by Kaleb. No hardware, no cloud, nothing pushed/merged/published.

## Method label: REWARD-ONLY constrained / style-regularised warm-start from R9hg it1100. NOT imitation, NOT AMP.
Why not imitation: no G1-retargeted, licensed human floor-getup reference exists in the workspace (research/lane_b_references.md,
STATUS.md: "No existing get-up task or G1 reference motion exists locally or in Isaac Lab 3.0.0 tasks"); the repo has no tracking/AMP
reward, discriminator or reference loader (missing capability, reported to parent). I did not download any motion dataset.

## References (primary; used as design guidance only, nothing transplanted)
- HoST, Huang et al., arXiv:2502.08378 (https://arxiv.org/html/2502.08378), code https://github.com/InternRobotics/HoST (MIT, per repo research notes; licence not re-verified here):
  action bounds, hip-roll/yaw and foot-distance style rewards, curriculum, smoothness. Prior in-repo port: 0 valid stands, severe hard-limit violations (not rerun).
- HumanUP, https://humanoid-getup.github.io/ , arXiv:2502.12152 (https://arxiv.org/html/2502.12152): discovery stage, then slowed reference tracking of its own rollouts; not human mocap.
- AMP, Peng et al., arXiv:2104.02180 (https://arxiv.org/abs/2104.02180): needs licensed floor-getup motion retargeted to this G1 (37 DOF here). Not available.
- Official HoST/HumanUP stacks differ in DOF/physics (PhysX vs Newton/MJWarp, compliant limits); compatibility verified rather than assumed.
- Cause of the "splits" is NOT asserted; baseline foot-spread / hip telemetry is measured first.

## Frozen baseline
ckpt/R9hg_it1100_BASELINE_frozen.pt  sha256 665a89f58b3a3d76a1... == checkpoints/R9hg_it1100_candidate.pt (original untouched, read-only copy).
Sets: DEV = recovery/sets/dev_v1.npz (tuning), HELDOUT = recovery/sets/heldout_v2.npz (frozen comparison), TEST = untouched.
Criteria v3 (EVAL_PROTOCOL.md), assist off, same env seed (eval fixes seed 1), same 256 poses.

## Budget (hard): <= 2 h aggregate GPU incl. evals/renders. Ledger: ledger.csv (every GPU job via gpu.sh).
Training cap (set before start): <= 150 PPO iterations @ 16384 envs warm-start, wall timeout 50 min, save every 25 its.
Early stop: nonfinite loss/reward, persistent hard-limit/self-collision rise, DEV/HELDOUT success regression at the it+50 check.
Decision rule: candidate only if full-v3 success >= baseline - noise (3-8/256) AND hard-limit steps not worse AND peak/final spread lower.

## Known missing capabilities (for the parent's cloud coding agent; I did not edit source)
1. Reference-tracking / AMP: no reference loader, no discriminator, no tracking reward, no motion data.
2. Hip-roll/hip-yaw style reward term (HoST-like) - only a foot-distance term (stance_excess, free 0.45 m) exists.
3. eval_recovery.py does not report peak/final foot spread or hip roll/yaw per step; diag_trace.py saves stance width per step and q_end only.
4. isaaclab__train/envs_list cannot see g1_recovery tasks (needs PYTHONPATH + --external_callback): MCP gap.
