"""Build media/progression_chart.png and media/architecture.png from results/. Needs matplotlib."""
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
BLUE, GREY, RED = "#2a6fb0", "#8a8f98", "#c0392b"

# ---- progression chart -------------------------------------------------------
tr = list(csv.DictReader(open("results/training_curves.csv")))
it = [int(r["iteration"]) for r in tr]; el = [float(r["mean_episode_length"]) for r in tr]
ev = [r for r in csv.DictReader(open("results/eval_summary.csv")) if r["id"].startswith("newton_policy")]
ei = [int(r["id"].split("_it")[1][:4]) for r in ev]; es = [int(r["survivors"]) for r in ev]
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.05]})
a.plot(it, el, color=BLUE, lw=1.4); a.set_ylim(0, 1050)
a.axhline(1000, color=GREY, ls=":", lw=1); a.text(20, 1008, "max 1000 steps (20 s)", color=GREY, fontsize=8)
a.annotate("971 at iteration 1499", (1499, el[-1]), (850, 600), arrowprops=dict(arrowstyle="->", color="k"), fontsize=9)
a.set_xlabel("training iteration"); a.set_ylabel("mean episode length (steps)")
a.set_title("A. Training log (random commands, 1024 envs)", fontsize=10, loc="left")
b.bar(range(len(ei)), es, color=BLUE, width=0.7)
for x, v in enumerate(es): b.text(x, v + 4, f"{v}", ha="center", fontsize=9)
b.set_xticks(range(len(ei))); b.set_xticklabels(ei); b.set_ylim(0, 290)
b.axhline(256, color=GREY, ls=":", lw=1); b.text(-0.4, 261, "256 envs", color=GREY, fontsize=8)
b.set_xlabel("checkpoint (training iteration)"); b.set_ylabel("envs surviving the 24 s script")
b.set_title("B. Scripted stand/walk/turn/stop test, pushes", fontsize=10, loc="left")
for ax in (a, b): ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Newton/MJWarp run, seed 1: one training run, one evaluation per checkpoint", fontsize=11, x=0.01, ha="left")
fig.tight_layout(); fig.savefig("media/progression_chart.png", dpi=130)

# ---- architecture diagram ----------------------------------------------------
fig, ax = plt.subplots(figsize=(14.8, 3.8)); ax.axis("off"); ax.set_xlim(0, 14.6); ax.set_ylim(-0.3, 3.6)
W, G = 2.15, 0.28
boxes = [
 ("Isaac Lab task", "Isaac-Velocity-Flat-G1\nUnitree G1, rewards,\nterminations, pushes,\nphysics=newton_mjwarp", GREY),
 ("Observations", "base lin/ang velocity,\nprojected gravity,\nvelocity command,\njoint pos/vel, last action", GREY),
 ("Policy training", "rsl_rl PPO,\nMLP 256-128-128\n1024 envs x 24 steps\n1500 iterations, seed 1", BLUE),
 ("Checkpoint", "model_<iteration>.pt\nsaved every 50\niterations", BLUE),
 ("Scripted evaluation", "scripts/eval_g1.py\n256 envs, 24 s\nstand|walk|turn|stop\npushes <= 0.5 m/s", BLUE),
 ("Metrics and videos", "results/*.json, *.csv\nscripts/record_*.py\n-> MP4 / GIF", BLUE),
]
for k, (t, d, c) in enumerate(boxes):
    x = 0.1 + k * (W + G)
    ax.add_patch(FancyBboxPatch((x, 0.7), W, 2.1, boxstyle="round,pad=0.03", fc="white", ec=c, lw=2))
    ax.text(x + W / 2, 2.5, t, ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(x + W / 2, 1.5, d, ha="center", va="center", fontsize=8)
    if k < len(boxes) - 1:
        ax.add_patch(FancyArrowPatch((x + W + 0.03, 1.75), (x + W + G - 0.03, 1.75), arrowstyle="-|>", mutation_scale=14, color="k"))
# training loop: actions go back to the task
ax.add_patch(FancyArrowPatch((0.1 + 2 * (W + G) + W / 2, 0.68), (0.1 + W / 2, 0.68), connectionstyle="arc3,rad=-0.12", arrowstyle="-|>", mutation_scale=14, color=BLUE))
ax.text(3.6, -0.15, "joint-position actions (50 Hz) go back to the task during training", ha="center", fontsize=8.5, color=BLUE)
ax.text(0.1, 3.3, "grey = upstream Isaac Lab (task, robot, observations); blue = what this project configured, ran and wrote", fontsize=9, color=GREY)
fig.savefig("media/architecture.png", dpi=130, bbox_inches="tight")
