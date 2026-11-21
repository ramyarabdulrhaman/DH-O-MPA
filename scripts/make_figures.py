"""All result figures, drawn only from results/*_test.csv and raw_test.pkl."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex  # noqa: E402

RES, FIG = ex.RES, os.path.join(ex.ROOT, "figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "savefig.dpi": 400,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5})
ALG = list(ex.ALGOS)
PAL = ["#1d4ed8", "#0d9488", "#dc2626", "#f97316", "#a855f7", "#eab308", "#64748b", "#db2777", "#111827"]
COL = dict(zip(ALG, PAL))
MK = dict(zip(ALG, "os^vD<>Px"))
INST = ex.TEST
LBL = {"rgg500@42": "RGG-500", "rgg1000@42": "RGG-1000", "rgg2000@42": "RGG-2000", "hospital@42": "Hospital-1000",
       "brightkite@42": "Brightkite-1000", "ba500@42": "BA-500", "ba1500@42": "BA-1500"}
pr = pd.read_csv(os.path.join(RES, "per_run_test.csv"))
main = pr[pr.group == "main"]
st = pd.read_csv(os.path.join(RES, "stats_test.csv"))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)


# Fig: convergence on the ablation instance
cv = pd.read_csv(os.path.join(RES, "convergence_test.csv"))
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for a in ALG:
    g = cv[cv.algorithm == a].groupby("evals").HV
    if g.ngroups == 0:
        continue
    m, s, n = g.mean(), g.std().fillna(0), g.count()
    ax.plot(m.index, m.values, color=COL[a], marker=MK[a], markevery=3, ms=3, lw=1.3, label=a)
    ax.fill_between(m.index, m - 1.96 * s / np.sqrt(n), m + 1.96 * s / np.sqrt(n), color=COL[a], alpha=0.12, lw=0)
ax.set_xlabel("Function evaluations"); ax.set_ylabel("Hypervolume")
ax.legend(fontsize=5.8, ncol=2, frameon=False)
save(fig, "fig_convergence.png")

# Fig: HV distributions
fig, axs = plt.subplots(2, 4, figsize=(7.16, 3.9))
for ax, inst in zip(axs.flat, INST):
    d = [main[(main.instance == inst) & (main.algorithm == a)].HV.values for a in ALG]
    bp = ax.boxplot(d, patch_artist=True, widths=0.65, flierprops=dict(ms=1.5), medianprops=dict(color="k", lw=0.8))
    for p_, a in zip(bp["boxes"], ALG):
        p_.set_facecolor(COL[a]); p_.set_alpha(0.75); p_.set_linewidth(0.5)
    ax.set_xticks(range(1, len(ALG) + 1)); ax.set_xticklabels(ALG, rotation=60, ha="right", fontsize=5.5)
    ax.set_title(LBL[inst], fontsize=7.5, weight="bold")
axs.flat[-1].axis("off")
fig.tight_layout()
save(fig, "fig_hv_boxplots.png")

# Fig: effect-size heatmap (A12 of DH-O-MPA vs each baseline, HV)
h = st[st.metric == "HV"].pivot_table(index="baseline", columns="instance", values="A12")[INST]
o = st[st.metric == "HV"].pivot_table(index="baseline", columns="instance", values="outcome", aggfunc="first")[INST]
fig, ax = plt.subplots(figsize=(3.5, 2.6))
im = ax.imshow(h.values, cmap="RdBu", vmin=0, vmax=1, aspect="auto")
ax.grid(False)
ax.set_xticks(range(len(INST))); ax.set_xticklabels([LBL[i] for i in INST], rotation=40, ha="right", fontsize=6)
ax.set_yticks(range(len(h.index))); ax.set_yticklabels(h.index, fontsize=6.5)
for i in range(h.shape[0]):
    for j in range(h.shape[1]):
        v = h.values[i, j]
        ax.text(j, i, f"{v:.2f}{'' if o.values[i, j] == '+' else ('*' if o.values[i, j] == '-' else '°')}",
                ha="center", va="center", fontsize=5.5, color="white" if abs(v - 0.5) > 0.3 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.04); cb.set_label("$\\hat A_{12}$ (DH-O-MPA vs. row)", fontsize=6.5)
save(fig, "fig_effect_heatmap.png")

# Fig: runtime vs size
fig, ax = plt.subplots(figsize=(3.5, 2.4))
sz = ["rgg500@42", "rgg1000@42", "rgg2000@42"]
xs = [ex.get_inst(i).n for i in sz]
for a in ALG:
    m = [main[(main.instance == i) & (main.algorithm == a)].runtime_s.mean() for i in sz]
    ax.plot(xs, m, marker=MK[a], color=COL[a], lw=1.2, ms=3.5, label=a)
ax.set_yscale("log"); ax.set_xlabel("Number of devices $N$ (RGG)"); ax.set_ylabel("Time per run (s)")
ax.legend(fontsize=5.6, ncol=2, frameon=False)
save(fig, "fig_runtime.png")

# Fig: trade-offs (trust vs latency, trust vs energy) on RGG-1000, run 1
fr = pd.read_csv(os.path.join(RES, "fronts_test.csv"))
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.7))
for a in ALG:
    d = fr[(fr.instance == ex.ABLATION_INST) & (fr.algorithm == a)]
    axs[0].scatter(d.latency_ms, d.trust, s=10, color=COL[a], marker=MK[a], alpha=0.8, lw=0, label=a)
    axs[1].scatter(d.energy_mJ, d.trust, s=10, color=COL[a], marker=MK[a], alpha=0.8, lw=0)
axs[0].set_xlabel("End-to-end latency (ms)"); axs[1].set_xlabel("Path energy (mJ)")
for ax in axs:
    ax.set_ylabel("Path trust $f_{Trust}$")
axs[0].legend(fontsize=5.6, ncol=3, frameon=False, loc="lower right")
fig.tight_layout()
save(fig, "fig_fronts.png")

# Fig: ablation
ab = pr[pr.group == "ablation"]
order = sorted(ab.algorithm.unique())
fig, ax = plt.subplots(figsize=(3.5, 2.5))
bp = ax.boxplot([ab[ab.algorithm == v].HV.values for v in order], patch_artist=True, widths=0.6,
                flierprops=dict(ms=1.5), medianprops=dict(color="k", lw=0.8))
for p_, v in zip(bp["boxes"], order):
    p_.set_facecolor("#1d4ed8" if v.startswith("V3") else "#93c5fd"); p_.set_linewidth(0.5)
ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels([v.split(" ")[0] for v in order])
ax.set_ylabel(f"HV ({LBL[ex.ABLATION_INST]}, 30 runs)")
save(fig, "fig_ablation.png")

# Fig: network map with best-trust feasible paths (run 1) on RGG-1000
raw = pd.read_pickle(os.path.join(RES, "raw_test.pkl"))
I = ex.get_inst(ex.ABLATION_INST)
fig, ax = plt.subplots(figsize=(3.5, 3.4))
ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
for u, v in I.G.edges():
    ax.plot(*zip(I.pos[u], I.pos[v]), color="#cbd5e1", lw=0.3, zorder=1)
low = I.trust < ex.T_LOW
ax.scatter(I.pos[~low, 0], I.pos[~low, 1], s=3, color="#94a3b8", zorder=2, lw=0)
ax.scatter(I.pos[low & ~I.mal, 0], I.pos[low & ~I.mal, 1], s=5, color="#fb923c", zorder=2, lw=0, label="low-trust relay")
ax.scatter(I.pos[I.mal, 0], I.pos[I.mal, 1], s=7, color="#dc2626", marker="x", lw=0.6, zorder=3, label="compromised")
for a, lw in (("MPA", 1.2), ("CPO", 1.2), ("NSGA-III", 1.2), ("DH-O-MPA", 2.0)):
    r = next(r for r in raw if r["instance"] == ex.ABLATION_INST and r["algorithm"] == a and r["seed"] == 0)
    fe = [x for x in r["archive"] if x[1] == 0]
    if not fe:
        continue
    p = min(fe, key=lambda x: x[0][1])[2]   # fastest feasible path in the archive
    ax.plot(I.pos[p, 0], I.pos[p, 1], color=COL[a], lw=lw, zorder=4, label=f"{a} (fastest feasible)")
ax.scatter(*I.pos[I.src], s=40, marker="*", color="k", zorder=5); ax.scatter(*I.pos[I.dst], s=30, marker="s", color="k", zorder=5)
ax.annotate("$v_1$", I.pos[I.src], xytext=(4, 4), textcoords="offset points", fontsize=7)
ax.annotate("$v_m$", I.pos[I.dst], xytext=(4, 4), textcoords="offset points", fontsize=7)
ax.legend(fontsize=5.3, frameon=True, loc="lower left", framealpha=0.9)
save(fig, "fig_network_map.png")
print("figures written")

# Fig: benchmark summary by regime (HV, feasibility, Friedman rank) -- data-driven
LONG, SHORT = ["rgg500@42", "rgg1000@42"], ["hospital@42", "brightkite@42", "ba500@42", "ba1500@42"]
fr = pd.read_csv(os.path.join(RES, "friedman_test.csv")).sort_values("mean_rank_HV")
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.5))
w = 0.8 / len(ALG)
for k, (met, ttl) in enumerate([("HV", "(a) Mean hypervolume"), ("feas", "(b) Feasibility rate (%)")]):
    ax = axs[k]
    for j, a in enumerate(ALG):
        vals = []
        for grp in (LONG, SHORT):
            d = main[(main.instance.isin(grp)) & (main.algorithm == a)]
            vals.append(d.HV.mean() if met == "HV" else 100 * (d.HV > 0).mean())
        ax.bar(np.arange(2) + (j - len(ALG) / 2 + 0.5) * w, vals, w, color=COL[a], label=a)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Long routes\n(RGG-500/1000)", "Short routes\n(Hosp., Bk, BA)"], fontsize=6.5)
    ax.set_title(ttl, fontsize=7.5)
h_, l_ = axs[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=9, fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.06))
ax = axs[2]
ax.bar(range(len(fr)), fr.mean_rank_HV, color=[COL[a] for a in fr.algorithm])
ax.set_xticks(range(len(fr))); ax.set_xticklabels(fr.algorithm, rotation=60, ha="right", fontsize=6)
ax.set_title("(c) Friedman mean rank (lower = better)", fontsize=7.5)
fig.tight_layout()
save(fig, "fig_benchmark_summary.png")
