"""
Week 7 - Step 7.4B.0
Very small diagnostic: locate trace-distance spikes in CONT.

Reads:
    results/07_04b_ancilla_joint_yx_timewise.csv

Outputs:
    results/07_04b0_trace_distance_spikes.csv
    results/07_04b0_trace_distance_vs_time.png
    results/07_04b0_spike_summary.txt

Purpose:
    Find WHEN the largest ancilla-measurement back-action events occur.
    No washout length is selected in this script.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = Path("results")
INPUT = RESULTS / "07_04b_ancilla_joint_yx_timewise.csv"

if not INPUT.exists():
    raise FileNotFoundError(
        f"Missing {INPUT}. Run 07_04b_ancilla_joint_yx_cont.py first."
    )

df = pd.read_csv(INPUT)

D = "trace_distance_pre_vs_post"
required = [
    "t",
    D,
    "purity_pre",
    "purity_post_nonselective",
    "ideal_yx",
    "p_plus",
    "p_minus",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

top = (
    df[required]
    .sort_values(D, ascending=False)
    .head(20)
    .reset_index(drop=True)
)

top.to_csv(
    RESULTS / "07_04b0_trace_distance_spikes.csv",
    index=False,
)

thresholds = [0.001, 0.01, 0.05, 0.10, 0.25]

summary_lines = []
summary_lines.append("WEEK 7 - STEP 7.4B.0 TRACE-DISTANCE SPIKE DIAGNOSTIC")
summary_lines.append("=" * 72)
summary_lines.append("")
summary_lines.append(f"Number of CONT time steps: {len(df)}")
max_idx = df[D].idxmax()
summary_lines.append(
    f"Maximum trace distance: {df.loc[max_idx, D]:.6f} "
    f"at t={int(df.loc[max_idx, 't'])}"
)
summary_lines.append(
    f"Median trace distance: {df[D].median():.6e}"
)
summary_lines.append("")

for th in thresholds:
    sel = df[df[D] > th]
    if len(sel) == 0:
        summary_lines.append(f"D > {th:.3f}: 0 time steps")
    else:
        tvals = sel["t"].astype(int).tolist()
        summary_lines.append(
            f"D > {th:.3f}: {len(sel)} time steps | "
            f"first t={tvals[0]} | last t={tvals[-1]} | "
            f"indices={tvals}"
        )

summary_lines.append("")
summary_lines.append("Top 20 spikes:")
summary_lines.append(top.to_string(index=False))

summary_text = "\n".join(summary_lines)

with open(
    RESULTS / "07_04b0_spike_summary.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(summary_text)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["t"], df[D], linewidth=1.0)
ax.set_xlabel("CONT time step t")
ax.set_ylabel("Trace distance D(rho, rho')")
ax.set_title("Step 7.4B.0 — Joint YX measurement back-action over time")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(
    RESULTS / "07_04b0_trace_distance_vs_time.png",
    dpi=180,
)
plt.close(fig)

print(summary_text)
print()
print("Saved:")
print("  results/07_04b0_trace_distance_spikes.csv")
print("  results/07_04b0_trace_distance_vs_time.png")
print("  results/07_04b0_spike_summary.txt")
