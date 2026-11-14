#!/usr/bin/env python3
"""Training curves and posture table for a check. Usage (lab python): plot_check.py <out_prefix> <label>=<jobid>[:<start_it>][@<eval.json>] ...
Writes <out_prefix>_curves.png (small multiples, one measure per panel, iterations since this run's start) and <out_prefix>_postures.png (postures reached on the eval set)."""
import sys, os, glob, json, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]   # dataviz reference categorical order, fixed per entity
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
LOGS = os.path.join(os.path.expanduser(os.environ.get("TWINSMITH_JOBS_DIR") or "~/.local/share/twinsmith/jobs"), "")
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF})
out = sys.argv[1]; runs = []
for a in sys.argv[2:]:
    lab, rest = a.split("=", 1); ev = None
    if "@" in rest: rest, ev = rest.split("@", 1)
    job, start = (rest.split(":") + ["0"])[:2]
    runs.append((lab, job, int(start), ev))
def run_dir(job):
    t = open(glob.glob(LOGS + job + "*/stdout.log")[0], errors="ignore").read().replace("\x00", "")
    ts = re.search(r"Exact experiment name requested from command line: (\S+)", t).group(1)
    return os.path.expanduser("~/robotics-workspace/envs/lab-3.0.0-src/logs/rsl_rl/g1_recovery_r/" + ts)
panels = [("Curriculum/lying_max_h", "Peak pelvis height from lying (m)", 0.74), ("Curriculum/lying_ever_supported", "Fraction reaching a supported stand", None), ("Episode_Reward/upright", "Upright reward per episode", None), ("Episode_Reward/height", "Height reward per episode", None)]
fig, axs = plt.subplots(2, 2, figsize=(11, 6.4), constrained_layout=True)
for ax, (tag, title, ref) in zip(axs.flat, panels):
    last = []
    for i, (lab, job, start, _) in enumerate(runs):
        ea = EventAccumulator(glob.glob(run_dir(job) + "/events*")[0], size_guidance={"scalars": 0}); ea.Reload()
        if tag not in ea.Tags()["scalars"]: continue
        s = ea.Scalars(tag); x = [p.step - s[0].step for p in s]; y = [p.value for p in s]
        ax.plot(x, y, color=PAL[i], lw=2, label=lab, solid_capstyle="round")
    if ref: ax.axhline(ref, color=MUTED, lw=1, ls=":"); ax.text(ax.get_xlim()[1], ref, " standing", va="center", ha="right", color=MUTED, fontsize=8)
    ax.set_title(title, loc="left", fontsize=11, color=INK); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_xlabel("iterations since this run started")
h, l = axs.flat[0].get_legend_handles_labels(); fig.legend(h, l, loc="outside lower center", ncol=len(runs), frameon=False)
fig.savefig(out + "_curves.png", dpi=130); print("wrote", out + "_curves.png")
ev = [(lab, json.load(open(e))) for lab, _, _, e in runs if e]
if ev:
    names = list(ev[0][1]["postures_reached"]); N = ev[0][1]["num_envs"]
    fig, axs = plt.subplots(1, len(names), figsize=(1.75 * len(names), 0.42 * len(ev) + 1.6), sharey=True, constrained_layout=True)
    for j, (ax, n) in enumerate(zip(axs, names)):
        for k, (lab, d) in enumerate(ev):
            v = d["postures_reached"][n]["envs"]; col = PAL[[r[0] for r in runs].index(lab)]
            ax.barh(len(ev) - 1 - k, v, color=col, height=0.72); ax.text(v + 6, len(ev) - 1 - k, str(v), va="center", fontsize=9, color=INK)
        ax.set_xlim(0, N * 1.28); ax.set_title(n.replace("_", "\n"), fontsize=9, loc="left"); ax.set_xticks([0, N]); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    axs[0].set_yticks(range(len(ev))); axs[0].set_yticklabels([lab for lab, _ in ev][::-1])
    fig.suptitle(f"Robots (of {N}) reaching each posture at least once, assist off, {ev[0][1].get('pose_set', '')}", x=0.01, ha="left", fontsize=11)
    fig.savefig(out + "_postures.png", dpi=130); print("wrote", out + "_postures.png")
