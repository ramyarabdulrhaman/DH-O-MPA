"""LaTeX tables + number macros generated from results/*_test.csv (no hand-typed numbers)."""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as ss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex  # noqa: E402

RES, TAB = ex.RES, os.path.join(ex.ROOT, "tables")
os.makedirs(TAB, exist_ok=True)
ALG = list(ex.ALGOS)
CITE = {"DH-O-MPA": r"\textbf{DH-O-MPA}", "MPA": r"MPA~\cite{faramarzi2020marine}", "NSGA-II": r"NSGA-II~\cite{deb2002fast}",
        "NSGA-III": r"NSGA-III~\cite{deb2014evolutionary}", "MOEA/D": r"MOEA/D~\cite{zhang2007moead}",
        "MOPSO": r"MOPSO~\cite{coello2004handling}", "ZOA": r"ZOA~\cite{trojovska2022zebra}",
        "CPO": r"CPO~\cite{abdelbasset2024crested}", "WS-KSP": r"WS-KSP~\cite{maristany2021improved}"}
INST = ex.TEST
LBL = {"rgg500@42": "RGG-500", "rgg1000@42": "RGG-1000", "rgg2000@42": "RGG-2000", "hospital@42": "Hosp.-1000",
       "brightkite@42": "Brightkite", "ba500@42": "BA-500", "ba1500@42": "BA-1500"}
s = pd.read_csv(os.path.join(RES, "summary_test.csv"))
sm = s[s.group == "main"].set_index(["instance", "algorithm"])
st = pd.read_csv(os.path.join(RES, "stats_test.csv"))
pr = pd.read_csv(os.path.join(RES, "per_run_test.csv"))
meta = pd.read_csv(os.path.join(RES, "instances_test.csv")).set_index("instance")
MARK = {"+": "$^{+}$", "-": "$^{-}$", "=": "$^{\\approx}$"}


def metric_table(met, hi, fmt, cap, label):
    L = [r"\begin{table*}[t]", r"\centering", r"\caption{" + cap + "}", r"\label{" + label + "}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "c" * len(INST) + "}", r"\toprule",
         "Algorithm & " + " & ".join(LBL[i] for i in INST) + r" \\", r"\midrule"]
    best = {i: (sm.loc[i][met].idxmax() if hi else sm.loc[i][met].idxmin()) for i in INST}
    for a in ALG:
        cells = []
        for i in INST:
            m, sd = sm.loc[(i, a), met], sm.loc[(i, a), met + "_std"]
            c = "--" if np.isnan(m) else fmt.format(m) + ("" if np.isnan(sd) else r"$\pm$" + fmt.format(sd))
            if a == best[i]:
                c = r"\textbf{" + c + "}"
            if a != "DH-O-MPA" and met in ("HV", "IGD"):
                q = st[(st.instance == i) & (st.metric == met) & (st.baseline == a)]
                if len(q):
                    c += MARK[q.outcome.iloc[0]]
            cells.append(c)
        L.append(CITE[a] + " & " + " & ".join(cells) + r" \\")
    if met in ("HV", "IGD"):
        L.append(r"\midrule")
        for sym, nm in (("+", "DH-O-MPA better (+)"), ("=", "no sig. difference ($\\approx$)"), ("-", "DH-O-MPA worse ($-$)")):
            cnt = [int(((st.instance == i) & (st.metric == met) & (st.outcome == sym)).sum()) for i in INST]
            L.append(nm + " & " + " & ".join(map(str, cnt)) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    open(os.path.join(TAB, label.split(":")[1] + ".tex"), "w").write("\n".join(L))


SIG = "Marks: $+$/$-$/$\\approx$ = DH-O-MPA significantly better/worse/no difference (Wilcoxon rank-sum, Holm-corrected, $\\alpha=0.05$; one-sample signed-rank vs.\\ the deterministic WS-KSP). Best mean in bold."
metric_table("HV", True, "{:.3f}", "Hypervolume (mean$\\pm$std, 30 runs, 3{,}000 evaluations; higher is better). " + SIG, "tab:hv")
metric_table("IGD", False, "{:.3f}", "IGD (mean$\\pm$std, 30 runs; lower is better). " + SIG, "tab:igd")
metric_table("runtime_s", False, "{:.1f}", "Wall-clock time per run in seconds (mean$\\pm$std).", "tab:runtime")

# instances table
L = [r"\begin{table}[t]", r"\centering", r"\caption{Test instances. $h$: hop distance $v_1\to v_m$; $H$: hop budget; $t_{\mathrm{dl}}$: deadline; Mal.: compromised relays; Low-T: share of relays with $T_{\mathrm{base}}<0.7$.}",
     r"\label{tab:instances}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
     r"Instance & $N$ & $|E|$ & $h$ & $H$ & $t_{\mathrm{dl}}$ (ms) & Mal. & Low-T \\", r"\midrule"]
for i in INST:
    m = meta.loc[i]
    L.append(f"{LBL[i]} & {int(m.nodes)} & {int(m.edges)} & {int(m.hop_dist)} & {int(m.hop_budget)} & {m.deadline_ms:.0f} & {int(m.n_malicious)} & {m.frac_low_trust:.2f}" + r" \\")
L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
open(os.path.join(TAB, "instances.tex"), "w").write("\n".join(L))

# extreme-objective table on large RGG + hospital
L = [r"\begin{table*}[t]", r"\centering",
     r"\caption{Best attainable value of each objective in the final feasible archive (mean over runs): highest path trust, lowest latency and lowest energy.}",
     r"\label{tab:extremes}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{ll" + "c" * len(ALG) + "}", r"\toprule",
     "Instance & Objective & " + " & ".join(CITE[a] for a in ALG) + r" \\", r"\midrule"]
for i in ["rgg1000@42", "rgg2000@42", "hospital@42"]:
    for met, nm, f, hi in (("max_trust", "max $f_{Trust}$", "{:.3f}", True), ("min_latency", "min latency (ms)", "{:.0f}", False),
                           ("min_energy_mJ", "min energy (mJ)", "{:.0f}", False)):
        v = sm.loc[i][met].reindex(ALG)
        b = v.idxmax() if hi else v.idxmin()
        cells = ["--" if np.isnan(v[a]) else ((r"\textbf{" + f.format(v[a]) + "}") if a == b else f.format(v[a])) for a in ALG]
        first = (r"\multirow{3}{*}{" + LBL[i] + "}") if met == "max_trust" else ""
        L.append(f"{first} & {nm} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule" if i != "hospital@42" else r"\bottomrule")
L += [r"\end{tabular}}", r"\end{table*}"]
open(os.path.join(TAB, "extremes.tex"), "w").write("\n".join(L))

# ablation
ab = pr[pr.group == "ablation"]
full = ab[ab.algorithm.str.startswith("V3")].HV.values
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{Ablation on RGG-1000 (30 runs). GD: guided decoder; AF: adaptive FADs; TPR: trust-guided path repair. $p$: Wilcoxon rank-sum vs.\ full DH-O-MPA (V3).}",
     r"\label{tab:ablation}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lccc}", r"\toprule",
     r"Variant & HV $\uparrow$ & IGD $\downarrow$ & $p$ (HV) \\", r"\midrule"]
for v in sorted(ab.algorithm.unique()):
    d = ab[ab.algorithm == v]
    p = "--" if v.startswith("V3") else f"{ss.mannwhitneyu(d.HV.values, full, alternative='two-sided').pvalue:.1e}"
    name = v.replace("->", r"$\to$")
    L.append(f"{name} & {d.HV.mean():.3f}$\\pm${d.HV.std():.3f} & {d.IGD.mean():.3f}$\\pm${d.IGD.std():.3f} & {p}" + r" \\")
L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
open(os.path.join(TAB, "ablation.tex"), "w").write("\n".join(L))

# hospital
h = pr[(pr.group == "main") & (pr.instance == "hospital@42")]
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{Hospital-1000 case study (30 runs): best-compromise path chosen by the same rule for all algorithms. Success: deadline met and no compromised relay.}",
     r"\label{tab:case}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lcccc}", r"\toprule",
     r"Algorithm & $f_{Trust}$ & Latency (ms) & Energy (mJ) & Success (\%) \\", r"\midrule"]
for a in ALG:
    d = h[h.algorithm == a]
    L.append(f"{CITE[a]} & {d.bc_trust.mean():.3f} & {d.bc_latency_ms.mean():.1f} & {d.bc_energy_mJ.mean():.0f} & {100 * d.success.astype(float).mean():.1f}" + r" \\")
L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
open(os.path.join(TAB, "case_study.tex"), "w").write("\n".join(L))

# macros
fr = pd.read_csv(os.path.join(RES, "friedman_test.csv"))
M = []
for i in INST:
    d = sm.loc[i].HV
    bb = d.drop("DH-O-MPA").idxmax()
    k = {"rgg500@42": "RGGa", "rgg1000@42": "RGGb", "rgg2000@42": "RGGc", "hospital@42": "Hosp", "brightkite@42": "Bk", "ba500@42": "BAa", "ba1500@42": "BAb"}[i]
    bm = d.drop(["DH-O-MPA", "WS-KSP"]).idxmax()
    M += [f"\\newcommand{{\\gm{k}}}{{{100 * (d['DH-O-MPA'] - d[bm]) / d[bm]:.1f}}}", f"\\newcommand{{\\bm{k}}}{{{bm}}}"]
    M += [f"\\newcommand{{\\gain{k}}}{{{100 * (d['DH-O-MPA'] - d[bb]) / d[bb]:.1f}}}", f"\\newcommand{{\\bb{k}}}{{{bb}}}"]
M.append(f"\\newcommand{{\\friedchi}}{{{fr.chi2.iloc[0]:.1f}}}")
mm, ee = f"{fr.p.iloc[0]:.1e}".split("e")
M.append(f"\\newcommand{{\\friedp}}{{{mm}\\times10^{{{int(ee)}}}}}")
for _, r in fr.iterrows():
    M.append(f"\\newcommand{{\\rank{''.join(ch for ch in r.algorithm if ch.isalpha())}}}{{{r.mean_rank_HV:.2f}}}")
hv = st[st.metric == "HV"]
M += [f"\\newcommand{{\\nwin}}{{{int((hv.outcome == '+').sum())}}}", f"\\newcommand{{\\ntie}}{{{int((hv.outcome == '=').sum())}}}",
      f"\\newcommand{{\\nloss}}{{{int((hv.outcome == '-').sum())}}}", f"\\newcommand{{\\ntot}}{{{len(hv)}}}"]
open(os.path.join(TAB, "numbers.tex"), "w").write("\n".join(M))
print("\n".join(M))
