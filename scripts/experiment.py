"""
DH-O-MPA v2 experiment suite (tuning/test protocol).

Protocol
  * TUNE instances (different graph seeds) are used ONLY to develop and select the
    DH-O-MPA configuration (python scripts/experiment.py --phase tune).
  * The selected configuration is frozen in FINAL_CONFIG below, and the TEST
    instances are run once with all algorithms (--phase test).
  * All algorithms share decoder, archive, constraint handling and evaluation budget.
  * NSGA-II, NSGA-III and MOEA/D are the reference implementations from pymoo,
    wrapped around the same decoder/evaluator.

Nothing in this file is hand-typed into result tables.
"""
import argparse
import math
import os
import time
import zlib
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

B_TASK, C_TASK = 1.0e5, 1.0e7
E_ELEC, EPS_AMP, GAMMA, KAPPA = 50e-9, 100e-12, 2.0, 1e-27
ALPHA_T, BETA_T = 0.7, 0.3
W_CL, W_CW, W_OL = 0.4, 0.4, 0.2
REQ_CPU, REQ_RAM, REQ_BW = 1.5, 2.0, 20.0
LB, UB = -5.0, 5.0
N_POP, N_ARCH = 30, 40
T_LOW = 0.70


# ============================================================ instance generators
def _attach(G, rng, pos, trust=None, lat_fn=None, mal=None):
    for v in G.nodes():
        d = G.nodes[v]
        d["pos"] = np.asarray(pos[v], float)
        d.setdefault("trust", float(trust[v]) if trust is not None else float(rng.uniform(0.5, 0.95)))
        d.setdefault("cpu", float(rng.uniform(1.0, 3.2)))
        d.setdefault("ram", float(rng.uniform(1.0, 8.0)))
        d.setdefault("battery", float(rng.uniform(0.05, 1.0)))
        d.setdefault("queue", float(rng.uniform(0.5, 3.0)))
        d["malicious"] = bool(mal is not None and v in mal)
    for u, v in G.edges():
        e = G.edges[u, v]
        dist = float(np.linalg.norm(G.nodes[u]["pos"] - G.nodes[v]["pos"]))
        e.setdefault("lat", float(lat_fn(dist, rng)) if lat_fn else float(rng.uniform(5, 25)))
        e.setdefault("bw", float(rng.uniform(10.0, 100.0)))
        e.setdefault("cl", float(rng.uniform(0.3, 0.9)))
        e.setdefault("cw", float(rng.uniform(0.1, 0.8)))
        e.setdefault("ol", 1.0 if rng.random() < 0.2 else 0.0)
        if G.nodes[u]["malicious"] or G.nodes[v]["malicious"]:
            e["lat"] *= 0.5     # black-hole lure: attackers advertise fast links
    return G


def build_ba(n, k, seed, lat=(10, 60), tlo=0.3):
    rng = np.random.default_rng(seed)
    G = nx.barabasi_albert_graph(n, k // 2, seed=seed)
    lay = nx.spring_layout(G, seed=seed)
    pos = {v: (np.asarray(p) + 1) * 250 for v, p in lay.items()}
    tr = {v: rng.uniform(tlo, 0.95) for v in G}
    return _attach(G, rng, pos, tr, lambda d, r: r.uniform(*lat))


def build_rgg(n, seed, deg=7.0, side=1000.0, zones=4, zone_r=0.12):
    """Random geometric SIoT deployment with spatially clustered low-trust zones."""
    rng = np.random.default_rng(seed)
    r = math.sqrt(deg / (math.pi * n))
    G0 = nx.random_geometric_graph(n, r, seed=int(seed))
    G0 = G0.subgraph(max(nx.connected_components(G0), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G0)
    P = {v: np.asarray(G.nodes[v]["pos"]) for v in G}
    centers = rng.random((zones, 2))
    tr, mal = {}, set()
    for v in G:
        in_zone = np.min(np.linalg.norm(centers - P[v], axis=1)) < zone_r
        if in_zone:
            tr[v] = rng.uniform(0.15, 0.55)
            if tr[v] < 0.35:
                mal.add(v)
        else:
            tr[v] = rng.beta(7, 2)
    for v in G:
        del G.nodes[v]["pos"]
    pos = {v: P[v] * side for v in G}
    rad = r * side
    return _attach(G, rng, pos, tr, lambda d, rr: rr.uniform(1, 4) + 12.0 * d / rad, mal)


def build_hospital_large(seed):
    """Multi-wing hospital: 1000 devices, gateway backbone as a geometric mesh, 40 compromised relays."""
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    W, H = 300.0, 120.0
    n_gw, n_sens, n_med, n_db = 160, 640, 180, 20
    pos, tr = {}, {}
    gws = list(range(n_gw))
    for g in gws:
        pos[g] = np.array([rng.uniform(0, W), rng.uniform(0, H)])
        G.add_node(g, cpu=float(rng.uniform(2.5, 4.0)), ram=float(rng.uniform(8, 32)), battery=1.0)
        tr[g] = rng.uniform(0.80, 0.99)
    gp = np.array([pos[g] for g in gws])
    for g in gws:
        d = np.linalg.norm(gp - pos[g], axis=1)
        for h in np.where((d > 0) & (d < 32.0))[0]:
            G.add_edge(g, int(h))
    nid = n_gw
    for kind, cnt in (("sensor", n_sens), ("med", n_med)):
        for _ in range(cnt):
            pos[nid] = np.array([rng.uniform(0, W), rng.uniform(0, H)])
            if kind == "sensor":
                G.add_node(nid, cpu=float(rng.uniform(0.2, 0.8)), ram=float(rng.uniform(0.1, 0.5)), battery=float(rng.uniform(0.05, 0.9)))
                tr[nid] = rng.uniform(0.70, 0.98)
            else:
                G.add_node(nid, cpu=float(rng.uniform(1.5, 2.5)), ram=float(rng.uniform(2, 8)), battery=float(rng.uniform(0.5, 1.0)))
                tr[nid] = rng.uniform(0.75, 0.98)
            d = np.linalg.norm(gp - pos[nid], axis=1)
            for h in np.argsort(d)[:3]:
                G.add_edge(nid, int(h))
            nid += 1
    dbs = []
    for _ in range(n_db):     # data centre at the east end
        pos[nid] = np.array([W + rng.uniform(5, 15), rng.uniform(0, H)])
        G.add_node(nid, cpu=4.5, ram=64.0, battery=1.0)
        tr[nid] = 0.99
        d = np.linalg.norm(gp - pos[nid], axis=1)
        for h in np.argsort(d)[:4]:
            G.add_edge(nid, int(h))
        dbs.append(nid)
        nid += 1
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    mal = set(int(x) for x in rng.choice([g for g in gws if g in G], 30, replace=False))
    mal |= set(int(x) for x in rng.choice([v for v in G if n_gw <= v < n_gw + n_sens + n_med], 10, replace=False))
    for m in mal:
        tr[m] = rng.uniform(0.15, 0.34)
    lat = lambda d, r: r.uniform(0.3, 1.2) + d / 12.0
    G = _attach(G, rng, pos, tr, lat, mal)
    return G, dbs


def build_brightkite(seed=42, n_nodes=1000):
    rng = np.random.default_rng(seed)
    G0 = nx.read_edgelist(os.path.join(ROOT, "data", "loc-brightkite_edges.txt.gz"), nodetype=int)
    cc = G0.subgraph(max(nx.connected_components(G0), key=len))
    start = max(cc.degree, key=lambda x: x[1])[0]
    G = nx.convert_node_labels_to_integers(nx.Graph(cc.subgraph(list(nx.bfs_tree(cc, start).nodes())[:n_nodes])))
    pos = {v: rng.uniform(0, 500, 2) for v in G}
    tr = {v: rng.beta(7, 2) for v in G}
    return _attach(G, rng, pos, tr, lambda d, r: r.uniform(10, 50))


# ============================================================ instance object
class Instance:
    def __init__(self, name, G, src, dst, t_min=0.35, e_cut=0.10, dl_factor=1.5, seed=7):
        self.name, self.G = name, G
        rng = np.random.default_rng(seed)
        n = G.number_of_nodes()
        assert set(G.nodes()) == set(range(n))
        self.n = n
        self.nbrs = [list(G.neighbors(v)) for v in range(n)]
        nd = G.nodes
        self.trust = np.array([nd[v]["trust"] for v in range(n)])
        self.cpu = np.array([nd[v]["cpu"] for v in range(n)])
        self.ram = np.array([nd[v]["ram"] for v in range(n)])
        self.bat = np.array([nd[v]["battery"] for v in range(n)])
        self.queue = np.array([nd[v]["queue"] for v in range(n)])
        self.mal = np.array([nd[v]["malicious"] for v in range(n)])
        self.pos = np.array([nd[v]["pos"] for v in range(n)])
        tdir = {}
        for u in range(n):
            for v in self.nbrs[u]:
                k = int(rng.integers(5, 30))
                s = int(rng.binomial(k, self.trust[v]))
                tdir[(u, v)] = (s + 1.0) / (k + 2.0)
        nset = [set(x) for x in self.nbrs]
        self.T, self.L, self.BW, self.HL, self.HE = {}, {}, {}, {}, {}
        for u, v in G.edges():
            e = G.edges[u, v]
            S = W_CL * e["cl"] + W_CW * e["cw"] + W_OL * e["ol"]
            for a, b in ((u, v), (v, u)):
                K = nset[a] & nset[b]
                tind = (sum(tdir[(a, k)] * tdir[(k, b)] for k in K) / (sum(tdir[(a, k)] for k in K) + 1e-6)) if K else tdir[(a, b)]
                self.T[(a, b)] = (ALPHA_T * tdir[(a, b)] + (1 - ALPHA_T) * tind + BETA_T * S) / (1 + BETA_T)
                self.L[(a, b)] = e["lat"]
                self.BW[(a, b)] = e["bw"]
                # full per-hop latency and energy (used by evaluator and decoder heuristics)
                self.HL[(a, b)] = e["lat"] + 1e3 * B_TASK / (e["bw"] * 1e6) + self.queue[b] + 1e3 * C_TASK / (self.cpu[b] * 1e9)
                d = float(np.linalg.norm(self.pos[a] - self.pos[b]))
                self.HE[(a, b)] = 2 * E_ELEC * B_TASK + EPS_AMP * B_TASK * d ** GAMMA + KAPPA * (self.cpu[b] * 1e9) ** 2 * C_TASK
        self.t_min, self.e_cut = t_min, e_cut
        self.mHL = float(np.mean(list(self.HL.values())))
        self.mHE = float(np.mean(list(self.HE.values())))
        self.kappa = 1.0
        self.set_endpoints(src, dst, dl_factor)

    def set_endpoints(self, src, dst, dl_factor):
        self.src, self.dst = src, dst
        hd = nx.single_source_shortest_path_length(self.G, dst)
        self.hd = np.full(self.n, 10 ** 6)
        for v, d in hd.items():
            self.hd[v] = d
        self.next_hop = np.full(self.n, -1)
        for v in range(self.n):
            c = [w for w in self.nbrs[v] if self.hd[w] == self.hd[v] - 1]
            if c:
                self.next_hop[v] = c[0]
        # deadline and hop budget are derived from the minimum-latency ADMISSIBLE path
        # (relays satisfying the trust and battery constraints), so every instance is feasible.
        ok = lambda v: v in (src, dst) or (self.trust[v] >= self.t_min and self.bat[v] >= self.e_cut)
        H = nx.DiGraph()
        for (a, b), w in self.HL.items():
            if ok(a) and ok(b):
                H.add_edge(a, b, w=w)
        p = nx.shortest_path(H, src, dst, weight="w")
        self.min_lat = sum(self.HL[h] for h in zip(p[:-1], p[1:]))
        self.t_deadline = dl_factor * self.min_lat
        h0 = max(int(self.hd[src]), len(p) - 1)
        self.budget_hops = h0 + max(3, int(math.ceil(0.3 * h0)))

    # ------------------------------------------------ shared decoder
    def decode(self, x, rng, guided=False, pref=None, start=None, banned=()):
        """guided=False: argmax random-key decoder (baselines).
        guided=True : roulette over sigmoid(x) * T^mu * HL^-nu * HE^-eta (pref=(mu,nu,eta))."""
        path = list(start) if start else [self.src]
        visited = set(path) | set(banned)
        u = path[-1]
        mu, nu, eta = pref if pref is not None else (1.5, 1.0, 0.0)
        while u != self.dst:
            hops = len(path) - 1
            if self.dst in self.nbrs[u]:
                path.append(self.dst)
                break
            cand = [w for w in self.nbrs[u] if w not in visited and hops + 1 + self.hd[w] <= self.budget_hops]
            if not cand:
                v = u
                while v != self.dst:
                    v = int(self.next_hop[v])
                    path.append(v)
                break
            c = np.array(cand)
            if guided == "roulette":   # v1 stochastic decoder (kept for the ablation)
                sig = 1.0 / (1.0 + np.exp(-x[c]))
                w_ = sig * np.array([self.T[(u, w)] ** mu * self.HL[(u, w)] ** (-nu) * self.HE[(u, w)] ** (-eta) for w in cand])
                s = w_.sum()
                v = int(c[rng.choice(len(c), p=w_ / s)]) if s > 0 and np.isfinite(s) else int(c[np.argmax(x[c])])
            elif guided:   # biased random-key decoder: argmax of key + preference-weighted link heuristic
                h = np.array([mu * math.log(max(self.T[(u, w)], 1e-9))
                              - nu * math.log(self.HL[(u, w)] / self.mHL)
                              - eta * math.log(self.HE[(u, w)] / self.mHE) for w in cand])
                v = int(c[np.argmax(x[c] + self.kappa * h)])
            else:
                v = int(c[np.argmax(x[c])])
            path.append(v)
            visited.add(v)
            u = v
        return loop_erase(path)

    def evaluate(self, path):
        hops = list(zip(path[:-1], path[1:]))
        tr = np.array([self.T[h] for h in hops])
        f_trust = float(np.exp(np.mean(np.log(np.maximum(tr, 1e-9)))))
        lat = float(sum(self.HL[h] for h in hops))
        en = float(sum(self.HE[h] for h in hops))
        p = np.array(path)
        q = (min(1.0, self.cpu[p].min() / REQ_CPU) + min(1.0, self.ram[p].min() / REQ_RAM)
             + min(1.0, min(self.BW[h] for h in hops) / REQ_BW)) / 3.0
        F = np.array([1.0 - f_trust, lat, en, 1.0 - q])
        inner = p[1:-1]
        cv = max(0.0, lat - self.t_deadline) / self.t_deadline
        if len(inner):
            cv += max(0.0, self.e_cut - self.bat[inner].min()) + max(0.0, self.t_min - self.trust[inner].min())
        return F, cv


def loop_erase(path):
    out, idx = [], {}
    for v in path:
        if v in idx:
            k = idx[v]
            for w in out[k + 1:]:
                idx.pop(w, None)
            out = out[:k + 1]
        else:
            idx[v] = len(out)
            out.append(v)
    return out


def _far_pair(G0, rng, quantile=0.9, t_min=0.35, e_cut=0.10):
    # endpoints are chosen inside the largest component of admissible relays, so a feasible route exists
    adm = [v for v in G0 if G0.nodes[v]["trust"] >= t_min and G0.nodes[v]["battery"] >= e_cut]
    Ga = G0.subgraph(adm)
    G = Ga.subgraph(max(nx.connected_components(Ga), key=len))
    nodes = sorted(G.nodes())
    src = int(rng.choice(nodes))
    d = nx.single_source_shortest_path_length(G, src)
    src = max(d, key=d.get)                         # periphery node
    d = nx.single_source_shortest_path_length(G, src)
    target_h = int(np.quantile(list(d.values()), quantile))
    far = [v for v, h in d.items() if h == target_h]
    return src, int(rng.choice(far))


def make_instance(name):
    kind, *rest = name.split("@")
    seed = int(rest[0]) if rest else 42
    rng = np.random.default_rng(seed + 99)
    if kind.startswith("rgg"):
        n = int(kind[3:])
        G = build_rgg(n, seed)
        s, t = _far_pair(G, rng)
        return Instance(name, G, s, t)
    if kind.startswith("ba"):
        n = int(kind[2:])
        G = build_ba(n, 8, seed)
        s, t = _far_pair(G, rng)
        return Instance(name, G, s, t)
    if kind == "brightkite":
        G = build_brightkite(seed)
        s, t = _far_pair(G, rng)
        return Instance(name, G, s, t)
    if kind == "hospital":
        G, dbs = build_hospital_large(seed)
        G = nx.convert_node_labels_to_integers(G, label_attribute="orig")
        orig = {G.nodes[v]["orig"]: v for v in G}
        db = orig[dbs[0]] if dbs[0] in orig else max(G.nodes)
        # west-most wearable sensor as source
        sens = [v for v in G if 160 <= G.nodes[v]["orig"] < 800 and not G.nodes[v]["malicious"]]
        src = min(sens, key=lambda v: G.nodes[v]["pos"][0])
        return Instance(name, G, src, db, dl_factor=1.3)
    raise ValueError(name)


# ============================================================ archive / dominance
def dominates(Fa, ca, Fb, cb):
    if ca == 0 and cb > 0:
        return True
    if ca > 0 and cb == 0:
        return False
    if ca > 0 and cb > 0:
        return ca < cb
    return bool(np.all(Fa <= Fb) and np.any(Fa < Fb))


def crowding(F):
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        o = np.argsort(F[:, j])
        rg = F[o[-1], j] - F[o[0], j] or 1.0
        d[o[0]] = d[o[-1]] = np.inf
        d[o[1:-1]] += (F[o[2:], j] - F[o[:-2], j]) / rg
    return d


class Archive:
    def __init__(self, cap=N_ARCH):
        self.cap, self.items = cap, []

    def add(self, F, cv, x, path):
        for G_, c_, _, p_ in self.items:
            if dominates(G_, c_, F, cv) or p_ == path or (c_ == cv and np.array_equal(G_, F)):
                return False
        self.items = [it for it in self.items if not dominates(F, cv, it[0], it[1])]
        self.items.append((F, cv, None if x is None else x.copy(), path))
        while len(self.items) > self.cap:
            Fm = np.array([it[0] for it in self.items])
            lo, hi = Fm.min(0), Fm.max(0)
            self.items.pop(int(np.argmin(crowding((Fm - lo) / np.where(hi > lo, hi - lo, 1)))))
        return True

    def leader(self, rng):
        xs = [it[2] for it in self.items if it[2] is not None]
        return xs[int(rng.integers(len(xs)))]

    def F(self):
        it = [i[0] for i in self.items if i[1] == 0]
        return np.array(it) if it else np.zeros((0, 4))


class Run:
    def __init__(self, inst, seed, budget, snap_every=150):
        self.inst, self.rng, self.budget = inst, np.random.default_rng(seed), budget
        self.arch, self.evals, self.snaps, self.snap_every = Archive(), 0, [], snap_every

    def progress(self):
        return min(1.0, self.evals / self.budget)

    def done(self):
        return self.evals >= self.budget

    def eval_path(self, x, path):
        F, cv = self.inst.evaluate(path)
        self.evals += 1
        self.arch.add(F, cv, x, path)
        if self.evals % self.snap_every == 0:
            self.snaps.append((self.evals, self.arch.F().copy()))
        return F, cv, path

    def evaluate(self, x, guided=False, pref=None):
        return self.eval_path(x, self.inst.decode(x, self.rng, guided=guided, pref=pref))


def better(F1, c1, F2, c2, rng):
    if dominates(F1, c1, F2, c2):
        return True
    if dominates(F2, c2, F1, c1):
        return False
    return rng.random() < 0.5


def levy(rng, size, beta=1.5):
    sig = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
           (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    return rng.normal(0, sig, size) / np.abs(rng.normal(0, 1, size)) ** (1 / beta)


# ============================================================ DH-O-MPA family
def agent_prefs(rng, n, scale=3.0):
    """Preference-diverse decoding: each agent gets a Dirichlet weight vector over
    (trust, latency, energy) mapped to decoder exponents (mu, nu, eta)."""
    W = rng.dirichlet(np.ones(3), n)
    return [tuple(scale * w) for w in W]


def trust_path_repair(R, x, path, pref):
    """TPR: remove the least-trusted relay (or, if all relays are trusted, the slowest hop)
    and rebuild the remainder of the path with the guided decoder; encode it in the keys."""
    I = R.inst
    inner = path[1:-1]
    if not inner:
        return None
    tr = I.trust[inner]
    if tr.min() < T_LOW:
        j = 1 + int(np.argmin(tr))
    else:
        hl = [I.HL[(a, b)] for a, b in zip(path[:-1], path[1:])][1:]
        j = 1 + int(np.argmax(hl[:len(inner)]))
    bad = path[j]
    newp = I.decode(x, R.rng, guided=True, pref=pref, start=path[:j], banned=(bad,))
    if newp == path:
        return None
    xn = x.copy()
    xn[bad] = LB
    on = np.array(newp[1:-1], int)
    if len(on):
        xn[on] = np.maximum(xn[on], 0.8 * UB)
    return xn, newp


def mpa(R, guided=False, adaptive_fads=False, tpr=False, prefs=False, obl=False):
    n = R.inst.n
    X = R.rng.uniform(LB, UB, (N_POP, n))
    PR = agent_prefs(R.rng, N_POP) if prefs else [None] * N_POP
    ev = [R.evaluate(x, guided, PR[i]) for i, x in enumerate(X)]
    P = 0.5
    while not R.done():
        p = R.progress()
        CF = (1 - p) ** (2 * p)
        Xn = X.copy()
        for i in range(N_POP):
            E = R.arch.leader(R.rng)
            Rr = R.rng.random(n)
            if p < 1 / 3:
                RB = R.rng.normal(size=n)
                Xn[i] = X[i] + P * Rr * (RB * (E - RB * X[i]))
            elif p < 2 / 3:
                if i < N_POP // 2:
                    RL = 0.05 * levy(R.rng, n)
                    Xn[i] = X[i] + P * Rr * (RL * (E - RL * X[i]))
                else:
                    RB = R.rng.normal(size=n)
                    Xn[i] = E + P * CF * (RB * (RB * E - X[i]))
            else:
                RL = 0.05 * levy(R.rng, n)
                Xn[i] = E + P * CF * (RL * (RL * E - X[i]))
        if adaptive_fads:
            D = float(np.mean(np.abs(X - X.mean(0)) / (X.max(0) - X.min(0) + 1e-6)))
            FADs = 0.1 + 0.3 * (1 - np.clip(D, 0, 1))
        else:
            FADs = 0.2
        for i in range(N_POP):
            if R.rng.random() < FADs:
                U = R.rng.random(n) < FADs
                Xn[i] = Xn[i] + CF * (LB + R.rng.random(n) * (UB - LB)) * U
            else:
                r = R.rng.random()
                a, b = R.rng.integers(N_POP, size=2)
                Xn[i] = Xn[i] + (FADs * (1 - r) + r) * (X[a] - X[b])
        Xn = np.clip(Xn, LB, UB)
        for i in range(N_POP):
            if R.done():
                break
            F, cv, path = R.evaluate(Xn[i], guided, PR[i])
            if tpr and not R.done():
                rep = trust_path_repair(R, Xn[i], path, PR[i] if PR[i] else (1.5, 1.0, 0.0))
                if rep is not None:
                    xo, po = rep
                    Fo, cvo, _ = R.eval_path(xo, po)
                    if dominates(Fo, cvo, F, cv) or (cvo <= cv and Fo[0] < F[0] and R.rng.random() < 0.5):
                        Xn[i], F, cv = xo, Fo, cvo
            if obl and not R.done():   # legacy Trust-OBL (v1), kept for the ablation
                inner = path[1:-1]
                if any(R.inst.trust[v] < T_LOW for v in inner):
                    xo = Xn[i].copy()
                    for v in inner:
                        xo[v] = (LB + UB - xo[v] - (UB - LB) * 0.5 * R.rng.random()) if R.inst.trust[v] < T_LOW \
                            else xo[v] + R.rng.random() * 0.1 * (UB - LB) * R.inst.trust[v]
                    xo = np.clip(xo, LB, UB)
                    Fo, cvo, _ = R.evaluate(xo, guided, PR[i])
                    if Fo[0] < F[0] or dominates(Fo, cvo, F, cv):
                        Xn[i], F, cv = xo, Fo, cvo
            if better(F, cv, ev[i][0], ev[i][1], R.rng):
                X[i], ev[i] = Xn[i], (F, cv, None)


# ============================================================ swarm baselines
def mopso(R, c1=1.5, c2=1.5):
    n = R.inst.n
    X = R.rng.uniform(LB, UB, (N_POP, n))
    ev = [R.evaluate(x) for x in X]
    V = np.zeros_like(X)
    Pb, Pev = X.copy(), list(ev)
    vmax = 0.2 * (UB - LB)
    while not R.done():
        w = 0.9 - 0.5 * R.progress()
        for i in range(N_POP):
            if R.done():
                break
            L_ = R.arch.leader(R.rng)
            V[i] = np.clip(w * V[i] + c1 * R.rng.random(n) * (Pb[i] - X[i]) + c2 * R.rng.random(n) * (L_ - X[i]), -vmax, vmax)
            X[i] = np.clip(X[i] + V[i], LB, UB)
            F, cv, _ = R.evaluate(X[i])
            if better(F, cv, Pev[i][0], Pev[i][1], R.rng):
                Pb[i], Pev[i] = X[i].copy(), (F, cv, None)


def zoa(R):
    n = R.inst.n
    X = R.rng.uniform(LB, UB, (N_POP, n))
    ev = [R.evaluate(x) for x in X]
    while not R.done():
        for i in range(N_POP):
            for phase in (1, 2):
                if R.done():
                    break
                I = R.rng.integers(1, 3)
                if phase == 1:
                    xn = X[i] + R.rng.random(n) * (R.arch.leader(R.rng) - I * X[i])
                elif R.rng.random() < 0.5:
                    xn = X[i] + 0.01 * (2 * R.rng.random(n) - 1) * (1 - R.progress()) * X[i]
                else:
                    xn = X[i] + R.rng.random(n) * (X[R.rng.integers(N_POP)] - I * X[i])
                xn = np.clip(xn, LB, UB)
                F, cv, _ = R.evaluate(xn)
                if better(F, cv, ev[i][0], ev[i][1], R.rng):
                    X[i], ev[i] = xn, (F, cv, None)


def cpo(R, Tf=0.8, alpha=0.2):
    n = R.inst.n
    X = R.rng.uniform(LB, UB, (N_POP, n))
    ev = [R.evaluate(x) for x in X]
    while not R.done():
        p = R.progress()
        gam = 2 * R.rng.random() * (1 - p) ** p
        for i in range(N_POP):
            if R.done():
                break
            xcp = R.arch.leader(R.rng)
            r1, r2, r3 = R.rng.integers(N_POP, size=3)
            U1 = R.rng.random(n) > R.rng.random()
            if R.rng.random() < R.rng.random():
                y = (X[i] + X[r1]) / 2
                if R.rng.random() < R.rng.random():
                    xn = X[i] + R.rng.normal() * np.abs(2 * R.rng.random() * xcp - y)
                else:
                    xn = (1 - U1) * X[i] + U1 * (y + R.rng.random() * (X[r1] - X[r2]))
            else:
                delta = 1 if R.rng.random() < 0.5 else -1
                if R.rng.random() < Tf:
                    xn = (1 - U1) * X[i] + U1 * (X[r1] + R.rng.random() * (X[r2] - X[r3]) - delta * gam * R.rng.random(n))
                else:
                    r4, r5 = R.rng.random(), R.rng.random()
                    xn = xcp + (alpha * (1 - r4) + r4) * (delta * xcp - X[i]) - r5 * delta * gam * R.rng.random(n) * (X[r1] - X[i])
            xn = np.clip(xn, LB, UB)
            F, cv, _ = R.evaluate(xn)
            if better(F, cv, ev[i][0], ev[i][1], R.rng):
                X[i], ev[i] = xn, (F, cv, None)


# ============================================================ pymoo reference baselines
def pymoo_algo(R, which):
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    from pymoo.util.ref_dirs import get_reference_directions

    penal = which == "MOEA/D"   # pymoo MOEA/D has no constraint handling -> static penalty

    class P(ElementwiseProblem):
        def __init__(s):
            super().__init__(n_var=R.inst.n, n_obj=4, n_ieq_constr=0 if penal else 1, xl=LB, xu=UB)

        def _evaluate(s, x, out, *a, **k):
            F, cv, _ = R.evaluate(np.asarray(x))
            out["F"] = F + (1e3 * cv if penal else 0.0)
            if not penal:
                out["G"] = [cv]

    if which == "NSGA-II":
        from pymoo.algorithms.moo.nsga2 import NSGA2
        alg = NSGA2(pop_size=N_POP)
    elif which == "NSGA-III":
        from pymoo.algorithms.moo.nsga3 import NSGA3
        alg = NSGA3(ref_dirs=get_reference_directions("das-dennis", 4, n_partitions=3), pop_size=N_POP)
    else:
        from pymoo.algorithms.moo.moead import MOEAD
        alg = MOEAD(get_reference_directions("das-dennis", 4, n_partitions=3), n_neighbors=8, prob_neighbor_mating=0.7)
    minimize(P(), alg, get_termination("n_eval", R.budget), seed=int(R.rng.integers(1 << 30)), verbose=False)


# ============================================================ deterministic heuristic baseline
def wsum_ksp(R, n_weights=20):
    """Weighted-sum k-shortest simple paths (Yen) over the additive objectives
    (-log trust, latency, energy) for a sweep of weight vectors; deterministic."""
    I = R.inst
    W = np.random.default_rng(0).dirichlet(np.ones(3), n_weights)
    W[:3] = np.eye(3)
    sc = np.array([np.mean([-math.log(max(t, 1e-9)) for t in I.T.values()]),
                   np.mean(list(I.HL.values())), np.mean(list(I.HE.values()))])
    k = max(1, R.budget // n_weights)
    for w in W:
        H = nx.Graph()
        for (a, b) in I.T:
            if a < b:
                c = (w[0] * -math.log(max(min(I.T[(a, b)], I.T[(b, a)]), 1e-9)) / sc[0]
                     + w[1] * max(I.HL[(a, b)], I.HL[(b, a)]) / sc[1] + w[2] * max(I.HE[(a, b)], I.HE[(b, a)]) / sc[2])
                H.add_edge(a, b, c=c)
        for j, p in enumerate(nx.shortest_simple_paths(H, I.src, I.dst, weight="c")):
            if j >= k or R.done():
                break
            R.eval_path(None, p)


# ============================================================ registry
# FROZEN on 2026-11-13 after the tuning phase (best mean HV on all 4 TUNE instances = variant A5)
FINAL_CONFIG = dict(guided=True, adaptive_fads=True, tpr=True)

ALGOS = {
    "DH-O-MPA": (mpa, FINAL_CONFIG),
    "MPA": (mpa, {}),
    "NSGA-II": (pymoo_algo, dict(which="NSGA-II")),
    "NSGA-III": (pymoo_algo, dict(which="NSGA-III")),
    "MOEA/D": (pymoo_algo, dict(which="MOEA/D")),
    "MOPSO": (mopso, {}),
    "ZOA": (zoa, {}),
    "CPO": (cpo, {}),
    "WS-KSP": (wsum_ksp, {}),
}
VARIANTS_TUNE = {   # design-space explored on TUNE instances only
    "A0 MPA": dict(),
    "A1 +guided decoder": dict(guided=True),
    "A1r +roulette decoder (v1)": dict(guided="roulette"),
    "A2 +preference-diverse decoding": dict(guided=True, prefs=True),
    "A3 +adaptive FADs": dict(guided=True, prefs=True, adaptive_fads=True),
    "A4 +TPR": dict(guided=True, prefs=True, adaptive_fads=True, tpr=True),
    "A5 full w/o prefs": dict(guided=True, adaptive_fads=True, tpr=True),
    "A6 full w/o TPR, +legacy OBL": dict(guided=True, prefs=True, adaptive_fads=True, obl=True),
    "A7 full w/o adaptive FADs": dict(guided=True, prefs=True, tpr=True),
}
VARIANTS_TEST = {   # ablation of the frozen configuration on the TEST ablation instance
    "V0 MPA": dict(),
    "V1 MPA+GD": dict(guided=True),
    "V2 MPA+GD+AF": dict(guided=True, adaptive_fads=True),
    "V3 DH-O-MPA (GD+AF+TPR)": dict(FINAL_CONFIG),
    "V4 w/o GD": dict(adaptive_fads=True, tpr=True),
    "V5 w/o AF": dict(guided=True, tpr=True),
    "V6 TPR->legacy OBL": dict(guided=True, adaptive_fads=True, obl=True),
    "V7 +preference decoding": dict(FINAL_CONFIG, prefs=True),
    "V8 GD->roulette (v1)": dict(guided="roulette", adaptive_fads=True, tpr=True),
}
VARIANTS = {**VARIANTS_TUNE, **VARIANTS_TEST}
TUNE = ["rgg300@1001", "rgg800@1002", "hospital@1003", "ba500@1004"]
TEST = ["rgg500@42", "rgg1000@42", "rgg2000@42", "hospital@42", "brightkite@42", "ba500@42", "ba1500@42"]
ABLATION_INST = "rgg1000@42"

_CACHE = {}


def get_inst(name):
    if name not in _CACHE:
        _CACHE[name] = make_instance(name)
    return _CACHE[name]


def task(args):
    inst_name, alg, group, seed, budget = args
    fn, kw = (ALGOS[alg] if group == "main" else (mpa, VARIANTS[alg]))
    inst = get_inst(inst_name)
    R = Run(inst, seed=seed * 1000 + zlib.crc32(alg.encode()) % 997, budget=budget)
    t0 = time.perf_counter()
    fn(R, **kw)
    return dict(instance=inst_name, algorithm=alg, group=group, seed=seed, runtime=time.perf_counter() - t0,
                evals=R.evals, archive=[(it[0], it[1], it[3]) for it in R.arch.items], snaps=R.snaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["tune", "test"], required=True)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    if a.phase == "tune":
        jobs = [(i, v, "ablation", s, a.budget) for i in TUNE for v in VARIANTS_TUNE for s in range(a.runs)]
        jobs += [(i, alg, "main", s, a.budget) for i in TUNE for alg in ("MPA", "CPO", "NSGA-III") for s in range(a.runs)]
    else:
        jobs = [(i, alg, "main", s, a.budget) for i in TEST for alg in ALGOS
                for s in range(1 if alg == "WS-KSP" else a.runs)]
        jobs += [(ABLATION_INST, v, "ablation", s, a.budget) for v in VARIANTS_TEST for s in range(a.runs)]
    # warm instance cache in parent for metadata
    print(f"{a.phase}: {len(jobs)} runs", flush=True)
    out, t0 = [], time.time()
    with Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(task, jobs, chunksize=1)):
            out.append(r)
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(RES, exist_ok=True)
    pd.to_pickle(out, os.path.join(RES, f"raw_{a.phase}.pkl"))
    meta = []
    for name in sorted({j[0] for j in jobs}):
        I = get_inst(name)
        meta.append(dict(instance=name, nodes=I.n, edges=I.G.number_of_edges(), src=I.src, dst=I.dst,
                         hop_dist=int(I.hd[I.src]), hop_budget=I.budget_hops, deadline_ms=round(I.t_deadline, 1),
                         min_latency_ms=round(I.min_lat, 1), n_malicious=int(I.mal.sum()),
                         frac_low_trust=round(float((I.trust < T_LOW).mean()), 3)))
    pd.DataFrame(meta).to_csv(os.path.join(RES, f"instances_{a.phase}.csv"), index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
