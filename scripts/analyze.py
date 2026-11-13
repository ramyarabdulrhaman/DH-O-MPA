"""Metrics/statistics from results/raw_<phase>.pkl.  Usage: python analyze.py tune|test"""
import os
import sys

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex  # noqa: E402

PH = sys.argv[1] if len(sys.argv) > 1 else "test"
RES = ex.RES
runs = pd.read_pickle(os.path.join(RES, f"raw_{PH}.pkl"))
hv = HV(ref_point=np.full(4, 1.1))


def feas(r):
    F = [a[0] for a in r["archive"] if a[1] == 0]
    return np.array(F) if F else np.zeros((0, 4))


norm = {}
for inst in sorted({r["instance"] for r in runs}):
    A = np.vstack([feas(r) for r in runs if r["instance"] == inst and len(feas(r))])
    P = np.unique(A[NonDominatedSorting().do(A, only_non_dominated_front=True)], axis=0)
    lo, hi = P.min(0), A.max(0)
    sc = np.where(hi > lo, hi - lo, 1.0)
    norm[inst] = (lo, sc, (P - lo) / sc)

rows = []
for r in runs:
    inst = r["instance"]
    F = feas(r)
    Fn = (F - norm[inst][0]) / norm[inst][1] if len(F) else F
    row = dict(instance=inst, algorithm=r["algorithm"], group=r["group"], seed=r["seed"],
               HV=float(hv(Fn)) if len(F) else 0.0, IGD=float(IGD(norm[inst][2])(Fn)) if len(F) else np.nan,
               front_size=len(F), runtime_s=r["runtime"],
               trust=1 - F[:, 0].mean() if len(F) else np.nan, latency_ms=F[:, 1].mean() if len(F) else np.nan,
               energy_mJ=1e3 * F[:, 2].mean() if len(F) else np.nan, qos=1 - F[:, 3].mean() if len(F) else np.nan,
               max_trust=1 - F[:, 0].min() if len(F) else np.nan, min_latency=F[:, 1].min() if len(F) else np.nan,
               min_energy_mJ=1e3 * F[:, 2].min() if len(F) else np.nan)
    if inst.startswith("hospital"):
        I = ex.get_inst(inst)
        fe = [a for a in r["archive"] if a[1] == 0]
        if fe:
            k = int(np.argmin(((np.array([a[0] for a in fe]) - norm[inst][0]) / norm[inst][1]).sum(1)))
            Fb, _, path = fe[k]
            row.update(bc_trust=1 - Fb[0], bc_latency_ms=Fb[1], bc_energy_mJ=1e3 * Fb[2],
                       bc_malicious=bool(I.mal[path].any()),
                       success=(Fb[1] <= I.t_deadline) and not I.mal[path].any())
        else:
            row.update(success=False)
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, f"per_run_{PH}.csv"), index=False)

cols = ["HV", "IGD", "trust", "latency_ms", "energy_mJ", "qos", "max_trust", "min_latency", "min_energy_mJ", "runtime_s", "front_size"]
g = df.groupby(["group", "instance", "algorithm"])[cols]
summ = g.mean().join(g.std(), rsuffix="_std").reset_index()
summ.to_csv(os.path.join(RES, f"summary_{PH}.csv"), index=False)


def a12(x, y):
    return ((x[:, None] > y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()) / (len(x) * len(y))


st = []
main = df[df.group == "main"]
algs = sorted(main.algorithm.unique())
if "DH-O-MPA" in algs:
    for inst in main.instance.unique():
        for met, hi in (("HV", True), ("IGD", False)):
            a = main[(main.instance == inst) & (main.algorithm == "DH-O-MPA")][met].dropna().values
            ps = []
            for alg in algs:
                if alg == "DH-O-MPA":
                    continue
                b = main[(main.instance == inst) & (main.algorithm == alg)][met].dropna().values
                if len(b) == 0:   # baseline returned no feasible solution at all
                    ps.append(dict(instance=inst, metric=met, baseline=alg, mean_dhompa=a.mean(), mean_baseline=np.nan,
                                   p_raw=0.0, A12=1.0, note="baseline found no feasible path"))
                    continue
                if len(b) < 2:   # deterministic single-run baseline: one-sample Wilcoxon signed-rank vs its value
                    p = stats.wilcoxon(a - b[0]).pvalue if np.any(a != b[0]) else 1.0
                    A = float(np.mean(a > b[0]) + 0.5 * np.mean(a == b[0])) if hi else float(np.mean(a < b[0]) + 0.5 * np.mean(a == b[0]))
                else:
                    p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
                    A = a12(a, b) if hi else a12(-a, -b)
                ps.append(dict(instance=inst, metric=met, baseline=alg, mean_dhompa=a.mean(), mean_baseline=b.mean(), p_raw=p, A12=A))
            o = np.argsort([x["p_raw"] for x in ps])
            prev = 0
            for rk, i in enumerate(o):
                prev = min(1.0, max(prev, (len(ps) - rk) * ps[i]["p_raw"]))
                ps[i]["p_holm"] = prev
            for x in ps:
                x["outcome"] = ("+" if x["A12"] > 0.5 else "-") if x["p_holm"] < 0.05 else "="
            st += ps
    pd.DataFrame(st).to_csv(os.path.join(RES, f"stats_{PH}.csv"), index=False)
    stoch = [a for a in algs if main[main.algorithm == a].seed.nunique() > 1]
    piv = main[main.algorithm.isin(stoch)].pivot_table(index=["instance", "seed"], columns="algorithm", values="HV")[stoch].dropna()
    fr = stats.friedmanchisquare(*[piv[c].values for c in stoch])
    rk = piv.rank(axis=1, ascending=False).mean()
    pd.DataFrame({"algorithm": rk.index, "mean_rank_HV": rk.values}).assign(chi2=fr.statistic, p=fr.pvalue) \
        .to_csv(os.path.join(RES, f"friedman_{PH}.csv"), index=False)

conv = []
for r in runs:
    if r["group"] == "main" and r["instance"] == ex.ABLATION_INST:
        for e, F in r["snaps"]:
            conv.append(dict(algorithm=r["algorithm"], seed=r["seed"], evals=e,
                             HV=float(hv((F - norm[r["instance"]][0]) / norm[r["instance"]][1])) if len(F) else 0.0))
pd.DataFrame(conv).to_csv(os.path.join(RES, f"convergence_{PH}.csv"), index=False)
fr_rows = [dict(instance=r["instance"], algorithm=r["algorithm"], trust=1 - F[0], latency_ms=F[1], energy_mJ=1e3 * F[2], qos=1 - F[3])
           for r in runs if r["group"] == "main" and r["seed"] == 0 for F in feas(r)]
pd.DataFrame(fr_rows).to_csv(os.path.join(RES, f"fronts_{PH}.csv"), index=False)

pd.set_option("display.width", 250)
p = summ.pivot_table(index=["group", "algorithm"], columns="instance", values="HV").round(3)
print(p.to_string())
if st:
    s = pd.DataFrame(st)
    print(s[s.metric == "HV"].pivot_table(index="baseline", columns="instance", values="outcome", aggfunc="first").to_string())
    print(s[s.metric == "IGD"].pivot_table(index="baseline", columns="instance", values="outcome", aggfunc="first").to_string())
    print(pd.read_csv(os.path.join(RES, f"friedman_{PH}.csv")).sort_values("mean_rank_HV").round(3).to_string())
