# Lane B status
- 17:0x wrote g1_recovery/lane_b.py (staged rewards, assist force via permanent_wrench_composer, global assist curriculum). Next: smoke test.
- smoke 10 it OK (4k steps/s @128 envs); probe: assist lifts pelvis (scale1.5 -> 0.47m); fixed eval-unassisted bug (training-only via fixed_poses_path None). Next: 30it/256 smoke + eval
