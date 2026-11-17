"""
LDA map T for correlation matrix memory keys.

Idea: find a linear map T so that keys whose recorded values are the same (class = distinct value id)
become correlated, while different classes stay decorrelated. Then the memory can store
M = sum_i T(k_i) u_i^T and a NEW key of a known class recalls its value (generalisation), which a
memory of orthogonal keys cannot do.

Where T acts matters
--------------------
  mode="composed"  T acts on the composed key (what CMM_OUT sees). Binding operations (PROD, SCC)
                   make keys quasi-orthogonal even when contexts share tokens, so a LINEAR map on
                   them is expected to find nothing: use it as the NULL CONTROL.
  mode="concat"    T acts on the pre-composition token vectors, concatenated: (h*d)-dim features.
                   This is the realistic test. The result is a similarity-preserving key
                   k' = normalize(T [x_1; ...; x_h]) that would replace the binding.
  mode="sum"       bag of tokens (d-dim), order-free.

Usage (ev = CMMSaturationEvaluator already built on the recorded store)
-----------------------------------------------------------------------
    from cmm_lda import run_lda_experiment
    res = run_lda_experiment(ev, n_train=200, n_test=20, modes=("composed", "concat"),
                             shrinks=(0.1, 0.5))

Every map is fitted on the first n_train batches (the stored pairs) and scored on the next n_test
batches, whose keys were never stored. Compare "lda_*" with the "random" map (same output size, same
pipeline): only a clear gap over "random" on held-out batches is generalisation; in-sample, LDA
always separates the classes (it fits noise when keys carry no class information).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# features and statistics
# --------------------------------------------------------------------------- #
def batch_features(store, i: int, device, mode: str = "concat", encode=None) -> torch.Tensor:
    """Features of the pairs of batch i (positions selected by to_save_sequences), flat order = b * n_save + j,
    i.e. the same order as CandidateSet.batch_ids[i]."""
    k = store.load_keys(i, device)                       # (B, L, h, d)
    ts = store.to_save(i)
    sel = list(ts) if ts is not None else None
    if mode == "composed":
        x = encode(k)                                    # (B, L, d)
        if sel is not None:
            x = x[:, sel]
        return x.reshape(-1, x.shape[-1])
    if sel is not None:
        k = k[:, sel]
    k = k.reshape(-1, k.shape[-2], k.shape[-1])          # (n, h, d)
    if mode == "concat":
        return k.reshape(k.shape[0], -1)                 # (n, h*d)
    if mode == "sum":
        return k.sum(1)                                  # (n, d)
    raise ValueError(f"unknown mode {mode}")


class ScatterAccumulator:
    """One-pass sufficient statistics for LDA: total second moment, class sums and class counts."""

    def __init__(self, p: int, n_classes: int, device):
        self.St = torch.zeros(p, p, device=device)
        self.S = torch.zeros(n_classes, p, device=device)
        self.n = torch.zeros(n_classes, device=device)
        self.N = 0

    @torch.no_grad()
    def add(self, X: torch.Tensor, y: torch.Tensor):
        self.St += X.T @ X
        self.S.index_add_(0, y, X)
        self.n.index_add_(0, y, torch.ones(len(y), device=X.device))
        self.N += X.shape[0]


# --------------------------------------------------------------------------- #
# maps
# --------------------------------------------------------------------------- #
@torch.no_grad()
def fit_lda(acc: ScatterAccumulator, m: int, shrink: float = 0.1, n_min: int = 2,
            balanced: bool = False) -> Dict:
    """
    Fisher LDA: maximise between-class over within-class scatter, S_b w = lambda S_w w.
      shrink    : S_w <- (1-shrink) S_w + shrink * mean(diag S_w) * I   (needed: S_w is badly conditioned)
      n_min     : classes with fewer samples do not contribute to S_b (their mean is too noisy)
      balanced  : weight classes equally in S_b instead of by frequency
    Returns dict(mu, W (p, m), evals); T(x) = (x - mu) W, then L2 normalisation.
    """
    N, S, n = acc.N, acc.S, acc.n
    p = acc.St.shape[0]
    dev = acc.St.device
    mu = S.sum(0) / N
    present = n > 0
    Sc, nc = S[present], n[present]

    # within-class scatter = sum_i x x^T - sum_c n_c mu_c mu_c^T
    Sw = (acc.St - (Sc / nc[:, None]).T @ Sc) / max(N - int(present.sum()), 1)
    Sw = (Sw + Sw.T) / 2

    use = nc >= n_min
    Mc = Sc[use] / nc[use, None] - mu
    w = torch.ones_like(nc[use]) if balanced else nc[use]
    Sb = (Mc * w[:, None]).T @ Mc / w.sum()

    eye = torch.eye(p, device=dev)
    Sw = (1 - shrink) * Sw + shrink * Sw.diagonal().mean() * eye
    L = torch.linalg.cholesky(Sw)
    Li = torch.linalg.solve_triangular(L, eye, upper=False)          # L^{-1}
    A = Li @ Sb @ Li.T
    evals, V = torch.linalg.eigh((A + A.T) / 2)                      # ascending
    m = max(1, min(m, int(use.sum()) - 1, p))
    W = Li.T @ V[:, -m:].flip(1)                                     # (p, m), W^T S_w W = I
    return dict(mu=mu, W=W, evals=evals[-m:].flip(0))


def random_map(p: int, m: int, device, seed: int = 0) -> Dict:
    """Control: random projection with the same output size."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(p, m, generator=g) / math.sqrt(p)
    return dict(mu=torch.zeros(p, device=device), W=W.to(device), evals=None)


def identity_map(p: int, device) -> Dict:
    """Control: no map (the plain memory)."""
    return dict(mu=torch.zeros(p, device=device), W=torch.eye(p, device=device), evals=None)


def transform(fit: Dict, X: torch.Tensor) -> torch.Tensor:
    return F.normalize((X - fit["mu"]) @ fit["W"], dim=-1)


# --------------------------------------------------------------------------- #
# held-out evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_maps(ev, fits: Dict[str, Dict], mode: str, train: range, test: range) -> Dict[str, Dict]:
    """
    For every map: build M = sum_i T(k_i) u_i^T and class centroids from the training batches, then, for the
    held-out pairs (keys never stored), measure
      cos_mean_seen        cosine(recall, true value)           (0 without generalisation)
      decode_acc_seen      nearest stored value to the recall   (cosine)
      centroid_acc_seen    nearest class centroid in key space  (classification by prototype)
      cos_own/other_centroid(s)  similarity of a held-out key to its own class centroid vs the others
    restricted to held-out pairs whose value occurs among the training pairs ("seen").
    """
    store, cands, dev = ev.store, ev.cands, ev.device
    V = cands.V
    Un = F.normalize(cands.vecs.to(dev), dim=-1)
    n_cls = torch.zeros(V, device=dev)
    state = {nm: dict(M=None, Cz=None) for nm in fits}

    for i in train:                                                  # ---- stored pairs
        X = batch_features(store, i, dev, mode, ev._encode)
        y = torch.as_tensor(cands.batch_ids[i], device=dev)
        U = cands.vecs[y.cpu()].to(dev)
        n_cls.index_add_(0, y, torch.ones(len(y), device=dev))
        for nm, f in fits.items():
            z = transform(f, X)
            st = state[nm]
            if st["M"] is None:
                st["M"] = torch.zeros(z.shape[1], U.shape[1], device=dev)
                st["Cz"] = torch.zeros(V, z.shape[1], device=dev)
            st["M"] += z.T @ U
            st["Cz"].index_add_(0, y, z)

    present = n_cls > 0
    n_present = int(present.sum())
    keys = ("cos", "own", "other", "nc_ok", "dec_ok", "seen")
    lists = {nm: {k: [] for k in keys} for nm in fits}

    for i in test:                                                   # ---- held-out pairs
        X = batch_features(store, i, dev, mode, ev._encode)
        y = torch.as_tensor(cands.batch_ids[i], device=dev)
        U = cands.vecs[y.cpu()].to(dev)
        seen = present[y]
        for nm, f in fits.items():
            z, st = transform(f, X), state[nm]
            out = z @ st["M"]
            sims = z @ F.normalize(st["Cz"], dim=-1).T               # (n, V)
            valid = sims.masked_fill(~present, 0.0)
            own = valid.gather(1, y[:, None]).squeeze(1)
            other = (valid.sum(1) - own) / max(n_present - 1, 1)
            nc_ok = sims.masked_fill(~present, -float("inf")).argmax(1) == y
            so = (F.normalize(out, dim=-1) @ Un.T).masked_fill(~present, -float("inf"))
            L = lists[nm]
            L["cos"].append(F.cosine_similarity(out, U, dim=-1).cpu())
            L["own"].append(own.cpu())
            L["other"].append(other.cpu())
            L["nc_ok"].append(nc_ok.float().cpu())
            L["dec_ok"].append((so.argmax(1) == y).float().cpu())
            L["seen"].append(seen.float().cpu())

    res = {}
    for nm, L in lists.items():
        c = {k: torch.cat(v) for k, v in L.items()}
        s = c["seen"] > 0
        mean = lambda t: float(t[s].mean()) if bool(s.any()) else float("nan")
        res[nm] = dict(
            m=int(state[nm]["M"].shape[0]),
            n_train_pairs=int(n_cls.sum()),
            frac_seen=float(c["seen"].mean()),
            cos_mean_seen=mean(c["cos"]),
            decode_acc_seen=mean(c["dec_ok"]),
            centroid_acc_seen=mean(c["nc_ok"]),
            cos_own_centroid=mean(c["own"]),
            cos_other_centroids=mean(c["other"]),
            chance=1.0 / n_present,
        )
    return res


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_lda_experiment(ev, n_train: int, modes: Sequence[str] = ("composed", "concat"),
                       m: Optional[int] = None, shrinks: Sequence[float] = (0.1, 0.5),
                       n_min: int = 2, balanced: bool = False, verbose: bool = True) -> Dict:
    """Fit LDA (several shrinkage values) plus controls on the first n_train recorded batches and score on the
    next n_test. Returns {mode: {map_name: metrics}}."""
    store, cands, dev = ev.store, ev.cands, ev.device
    if n_train > len(store):
        raise ValueError(f"need n_train + n_test <= {len(store)} recorded batches")
    train, test = range(n_train), range(n_train)

    results = {}
    for mode in modes:
        p = batch_features(store, 0, dev, mode, ev._encode).shape[1]
        acc = ScatterAccumulator(p, cands.V, dev)
        for i in train:                                              # one pass: sufficient statistics
            acc.add(batch_features(store, i, dev, mode, ev._encode),
                    torch.as_tensor(cands.batch_ids[i], device=dev))

        m_eff = min(m or ev.layer.d, p)
        fits = {"random": random_map(p, m_eff, dev)}
        if mode == "composed":
            fits["identity"] = identity_map(p, dev)
        for sh in shrinks:
            fits[f"lda_shrink{sh}"] = fit_lda(acc, m_eff, shrink=sh, n_min=n_min, balanced=balanced)
        del acc

        res = evaluate_maps(ev, fits, mode, train, test)
        results[mode] = res
        if verbose:
            for nm, r in res.items():
                print(f"[{mode}] {nm:14s} m={r['m']:5d} seen={r['frac_seen']:.2f}  "
                      f"cos={r['cos_mean_seen']:.3f}  decode={r['decode_acc_seen']:.3f}  "
                      f"centroid={r['centroid_acc_seen']:.3f}  "
                      f"cos(own/other centroid)={r['cos_own_centroid']:.3f}/{r['cos_other_centroids']:.3f}  "
                      f"(chance {r['chance']:.1e})")
    return results