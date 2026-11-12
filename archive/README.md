# Archive: historical experiment data

This directory holds the raw material from the **recovery (get-up) experiments of 2026-11-06 to 2026-11-09**, before
the work was reorganised into the `walking/`, `recovery/` and `pilots/` trees. It is kept for provenance and
reproducibility; it is not the current result set.

| Path | Contents |
|---|---|
| `media/` | 6 dated clips and contact sheets (HoST and chain checks). The other get-up clips and sheets from this search were byte-identical to files in `recovery/media/` and are only kept there. |
| `results/` | 134 per-run evaluation JSONs, training summaries and the `linkcheck/` report (upper-body reference check). Walk reference: `results/walk_ref_limits.json`. |
| `runs/` | 87 training/evaluation logs, memory samples, and traces from the same runs. |

Where to look instead:

- **Current get-up write-up and curated videos:** [`recovery/README.md`](../recovery/README.md)
- **Current checkpoints and baselines:** [`recovery/checkpoints/README.md`](../recovery/checkpoints/README.md)
- **Current status:** [`STATUS.md`](../STATUS.md)
- **Dated experiments with their own conclusions:** [`pilots/`](../pilots/README.md)

The paths inside this directory are stable: experiment logs and reviews reference them as `archive/results/...`,
`archive/runs/...` and `archive/media/...`. New ad-hoc run logs and evaluation JSONs may also be written here by the
old `recovery/scripts/` helpers; curated outputs belong in `recovery/` or the relevant `pilots/` directory.
