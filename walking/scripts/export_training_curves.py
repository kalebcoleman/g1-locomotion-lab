"""Export scalar curves from an rsl_rl TensorBoard event file to CSV.
Usage: python export_training_curves.py <events file> <out.csv>"""
import sys, csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
TAGS = {"Train/mean_reward": "mean_reward", "Train/mean_episode_length": "mean_episode_length",
        "Episode_Termination/time_out": "frac_time_out", "Episode_Termination/base_contact": "frac_base_contact"}
ea = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0}); ea.Reload()
cols = {c: {s.step: s.value for s in ea.Scalars(t)} for t, c in TAGS.items()}
with open(sys.argv[2], "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["iteration"] + list(TAGS.values()))
    for it in sorted(cols["mean_reward"]):
        w.writerow([it] + [round(cols[c][it], 4) for c in TAGS.values()])
