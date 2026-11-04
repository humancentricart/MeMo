"""
Saturation evaluation for MeMoLayer correlation matrix memories, using the
REAL keys/values produced upstream (input_sequence / output_symbols).

Disk layout (one folder per layer, one file pair per memorize() call)
---------------------------------------------------------------------
    root/layer_{l}/batch_000000.keys.pt   input_sequence  (B, L, h, d)   shapes untouched
    root/layer_{l}/batch_000000.vals.pt   output_symbols  (B, L, d)
    root/layer_{l}/manifest.jsonl         one line per batch: shapes, to_save_sequences, n_pairs, is_last
    root/layer_{l}/candidates.pt          (created on first decoding run) distinct value vectors + ids

1) RECORD (your pipeline stays as is; memorize() is wrapped, not modified)
--------------------------------------------------------------------------
    rec = BatchRecorder("memo_dump", max_batches_per_layer=200)
    for l, layer in enumerate(self.layers):
        rec.attach(layer, l)
    ...  # run your normal loop: self.layers[l].memorize(input_sequence, output_symbols,
         #                                  to_save_sequences=range(start, L), is_last=...)
    rec.detach_all()

   (or call rec.record(layer_level, input_sequence, output_symbols, to_save_sequences, is_last)
    explicitly, wherever you prefer)

2) REPLAY + EVALUATE (per layer, possibly in another process)
-------------------------------------------------------------
    store = BatchStore("memo_dump", layer_level=2)
    ev = CMMSaturationEvaluator(layer, store, n_holdout_batches=2)
    rows = ev.sweep(default_checkpoints(ev.max_train_batches, n_points=8),
                    n_eval=2000, n_fresh=2000, topk=(1, 5, 10, 100),
                    out_json="memo_dump/layer2_sweep.json")
    plot_sweep(rows, "memo_dump/layer2_sweep.png")

A "pair" is one (key, value) at a position selected by to_save_sequences, so one
batch holds B * len(to_save_sequences) pairs. The last `n_holdout_batches` batches are
never memorized; they provide unseen keys to measure pure crosstalk.

3) DECODERS (new; on by default, disable with decode=False)
-----------------------------------------------------------
The candidates ("vocabulary") are the DISTINCT value vectors found in the whole store
(identified by exact equality); a decoder turns the recall r = M k into a distribution over
them. Compared, per checkpoint:

    dot_T1          logits = r . u_c                               (plain unembedding, T = 1)
    bayes_T         logits = beta * (r . u_c - 0.5*g*|u_c|^2)      beta from theory (see below)
    bayes_T_prior   bayes_T + log pi_c                             pi_c = smoothed stored frequency
    fit_T           same as bayes_T, beta fitted by max likelihood (2-fold cross-fitting)
    fit_T_prior     same as bayes_T_prior, beta fitted

Theory: for unit-norm keys and isotropic values, r = g*u_true + N(0, s^2 I) with
s^2 = ((P-1)/d) * sigma_v^2 (sigma_v^2 = per-component value variance), g = |k|^2 (~1), hence
beta = g / s^2. The temperature only changes calibration (NLL), never the arg-max; only the prior
(or the |u_c|^2 bias when candidate norms differ) can change the ranking.
Whitening with the value covariance is NOT included: it drops out when values are isotropic.

prior_smoothing = a in pi_c = (count_c + a) / (P + a*V). a = 1 is Laplace smoothing (unseen
candidates keep some mass); a -> 0 restricts the decoder to candidates that were actually stored.

Top-k accuracy
--------------
`topk` (evaluate / sweep) is a list of k values, e.g. (1, 5, 10, 100). For every decoder the
result dict contains "top{k}" for each k (k = 1 is always included). If k exceeds the number of
candidates V, it is clipped to V (accuracy is then trivially 1).
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

DECODER_NAMES = ("dot_T1", "bayes_T", "bayes_T_prior", "fit_T", "fit_T_prior")
DEFAULT_TOPK = (1, 5, 10, 100)


def _norm_ks(topk: Sequence[int]) -> List[int]:
    """Sorted, de-duplicated list of k values; k = 1 is always included."""
    if isinstance(topk, int):
        topk = [topk]
    ks = sorted({int(k) for k in topk} | {1})
    if ks[0] < 1:
        raise ValueError(f"all k in topk must be >= 1, got {list(topk)}")
    return ks


# --------------------------------------------------------------------------- #
# to_save_sequences (de)serialization: None | range | list/tensor of indices
# --------------------------------------------------------------------------- #
def _encode_to_save(ts):
    if ts is None:
        return None
    if isinstance(ts, range):
        return {"type": "range", "start": ts.start, "stop": ts.stop, "step": ts.step}
    return {"type": "list", "values": [int(i) for i in torch.as_tensor(ts).flatten().tolist()]}


def _decode_to_save(d):
    """Return exactly what the layer originally received (None, range or list)."""
    if d is None:
        return None
    if d["type"] == "range":
        return range(d["start"], d["stop"], d["step"])
    return d["values"]


def _n_selected(d, L: int) -> int:
    if d is None:
        return L
    if d["type"] == "range":
        return len(range(d["start"], d["stop"], d["step"]))
    return len(d["values"])


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #
class BatchRecorder:
    """
    Saves, at every memorize() invocation, the exact input_sequence, output_symbols
    and to_save_sequences, one file pair per call, grouped by layer level.
    Resumable: if the folder already has batches, numbering continues.
    """

    def __init__(self, root, save_dtype: Optional[torch.dtype] = None,
                 max_batches_per_layer: Optional[int] = None):
        self.root = Path(root)
        self.save_dtype = save_dtype  # e.g. torch.float16 to halve disk (replay is then lossy)
        self.max_batches = max_batches_per_layer
        self._counters: Dict[int, int] = {}
        self._originals: Dict[int, Tuple[torch.nn.Module, object]] = {}

    # ---- internals --------------------------------------------------------
    def _layer_dir(self, level: int) -> Path:
        d = self.root / f"layer_{level}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _next_idx(self, level: int) -> int:
        if level not in self._counters:
            m = self._layer_dir(level) / "manifest.jsonl"
            n = 0
            if m.exists():
                with open(m) as f:
                    n = sum(1 for line in f if line.strip())
            self._counters[level] = n
        return self._counters[level]

    def _pack(self, x: torch.Tensor) -> torch.Tensor:
        # clone() so torch.save writes only this tensor, not a larger underlying storage
        x = x.detach().cpu().contiguous().clone()
        if self.save_dtype is not None and x.is_floating_point():
            x = x.to(self.save_dtype)
        return x

    # ---- public API -------------------------------------------------------
    @torch.no_grad()
    def record(self, layer_level: int, input_sequence: torch.Tensor, output_symbols: torch.Tensor,
               to_save_sequences=None, is_last: bool = False) -> Optional[int]:
        i = self._next_idx(layer_level)
        if self.max_batches is not None and i >= self.max_batches:
            return None
        B, L = input_sequence.shape[:2]
        ts = _encode_to_save(to_save_sequences)
        n_pairs = B * _n_selected(ts, L)
        if n_pairs == 0:  # nothing would be memorized
            return None

        d = self._layer_dir(layer_level)
        k, v = self._pack(input_sequence), self._pack(output_symbols)
        torch.save(k, d / f"batch_{i:06d}.keys.pt")
        torch.save(v, d / f"batch_{i:06d}.vals.pt")
        entry = dict(idx=i, keys_shape=list(k.shape), vals_shape=list(v.shape),
                     dtype=str(input_sequence.dtype), saved_dtype=str(k.dtype),
                     to_save=ts, n_pairs=n_pairs, is_last=bool(is_last))
        with open(d / "manifest.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        self._counters[layer_level] = i + 1
        return i

    def attach(self, layer, layer_level: int):
        """Wrap layer.memorize so every call is recorded; the call site is unchanged."""
        if layer_level in self._originals:
            raise RuntimeError(f"layer {layer_level} already attached")
        original = layer.memorize
        recorder = self

        @functools.wraps(original)
        def wrapped(input_sequence, output_symbols, to_save_sequences=None, is_last=False):
            recorder.record(layer_level, input_sequence, output_symbols, to_save_sequences, is_last)
            return original(input_sequence, output_symbols,
                            to_save_sequences=to_save_sequences, is_last=is_last)

        layer.memorize = wrapped
        self._originals[layer_level] = (layer, original)

    def detach_all(self):
        for layer, _ in self._originals.values():
            if "memorize" in layer.__dict__:
                del layer.memorize
        self._originals.clear()


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
class BatchStore:
    """Read-only access, batch by batch, to what BatchRecorder saved for one layer."""

    def __init__(self, root, layer_level: int):
        self.dir = Path(root) / f"layer_{layer_level}"
        m = self.dir / "manifest.jsonl"
        if not m.exists():
            raise FileNotFoundError(m)
        self.manifest: List[Dict] = [json.loads(l) for l in m.read_text().splitlines() if l.strip()]
        if not self.manifest:
            raise ValueError(f"{m} is empty")

    def __len__(self) -> int:
        return len(self.manifest)

    def n_pairs(self, i: int) -> int:
        return self.manifest[i]["n_pairs"]

    def to_save(self, i: int):
        return _decode_to_save(self.manifest[i]["to_save"])

    def is_last(self, i: int) -> bool:
        return self.manifest[i]["is_last"]

    @staticmethod
    def _cast(x, device, dtype):
        return x.to(device=device, dtype=dtype) if x.is_floating_point() else x.to(device)

    def load_keys(self, i, device="cpu", dtype=torch.float32):
        x = torch.load(self.dir / f"batch_{i:06d}.keys.pt", map_location="cpu")
        return self._cast(x, device, dtype)  # (B, L, h, d)

    def load_values(self, i, device="cpu", dtype=torch.float32):
        x = torch.load(self.dir / f"batch_{i:06d}.vals.pt", map_location="cpu")
        return self._cast(x, device, dtype)  # (B, L, d)

    def load(self, i, device="cpu", dtype=torch.float32):
        return (self.load_keys(i, device, dtype), self.load_values(i, device, dtype),
                self.to_save(i), self.is_last(i))


def default_checkpoints(max_batches: int, n_points: int = 8) -> List[int]:
    """Geometrically spaced numbers of batches, from 1 up to max_batches."""
    pts = np.unique(np.geomspace(1, max_batches, n_points).round().astype(int))
    return [int(p) for p in pts]


# --------------------------------------------------------------------------- #
# Candidate set (the "vocabulary" a decoder chooses among)
# --------------------------------------------------------------------------- #
class CandidateSet:
    """
    Distinct value vectors of the whole store (exact float equality) and, for every recorded
    pair (in the same flat order as CMMSaturationEvaluator._pairs), the id of its value.
    Built once per store and cached in <layer dir>/candidates.pt.
    """

    def __init__(self, store: BatchStore, cache: bool = True, max_cache_bytes: int = 3_000_000_000):
        self.max_cache_bytes = max_cache_bytes
        self._dev = None
        self._dev_sq = None
        self._dev_key = None

        path = store.dir / "candidates.pt"
        loaded = False
        if cache and path.exists():
            st = torch.load(path, map_location="cpu")
            if st["n_batches"] == len(store):
                self.vecs = st["vecs"]
                self.batch_ids = [t.numpy() for t in st["batch_ids"]]
                loaded = True
        if not loaded:
            self._build(store)
            if cache:
                torch.save(dict(vecs=self.vecs, n_batches=len(store),
                                batch_ids=[torch.from_numpy(a) for a in self.batch_ids]), path)
        self.V = self.vecs.shape[0]

    def _build(self, store: BatchStore):
        table: Dict[bytes, int] = {}
        rows: List[np.ndarray] = []
        batch_ids: List[np.ndarray] = []
        for i in range(len(store)):
            v = store.load_values(i, "cpu", torch.float32)
            ts = store.to_save(i)
            if ts is not None:
                v = v[:, list(ts)]
            v = np.ascontiguousarray(v.reshape(-1, v.shape[-1]).numpy())
            bi = np.empty(v.shape[0], dtype=np.int64)
            for r in range(v.shape[0]):
                h = hashlib.blake2b(v[r].tobytes(), digest_size=16).digest()
                j = table.get(h)
                if j is None:
                    j = len(rows)
                    table[h] = j
                    rows.append(v[r].copy())
                bi[r] = j
            batch_ids.append(bi)
        self.vecs = torch.from_numpy(np.stack(rows))
        self.batch_ids = batch_ids

    def chunks(self, device, dtype, chunk: int):
        """Yield (start, U[c, d], |U|^2[c]) on `device`; keeps a device copy if it is small enough."""
        if self.vecs.numel() * 4 <= self.max_cache_bytes:
            key = (str(device), dtype)
            if self._dev_key != key:
                self._dev = self.vecs.to(device=device, dtype=dtype)
                self._dev_sq = self._dev.pow(2).sum(-1)
                self._dev_key = key
            for s in range(0, self.V, chunk):
                yield s, self._dev[s:s + chunk], self._dev_sq[s:s + chunk]
        else:
            for s in range(0, self.V, chunk):
                U = self.vecs[s:s + chunk].to(device=device, dtype=dtype)
                yield s, U, U.pow(2).sum(-1)


class _TopK:
    """Streaming top-k over candidate chunks. Requires k <= total number of candidates."""

    def __init__(self, n: int, k: int, device):
        self.k = k
        self.v = torch.full((n, k), -float("inf"), device=device)
        self.i = torch.zeros((n, k), dtype=torch.long, device=device)

    def update(self, scores: torch.Tensor, offset: int):
        v, i = scores.topk(min(self.k, scores.shape[1]), dim=1)
        v = torch.cat([self.v, v], 1)
        i = torch.cat([self.i, i + offset], 1)
        v, j = v.topk(self.k, dim=1)
        self.v, self.i = v, i.gather(1, j)


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #
class CMMSaturationEvaluator:
    """
    Replays recorded batches into layer.CMM_OUT via layer.memorize(...) with the original
    arguments, then measures recall. Recall = CMM_OUT(encoded_key), i.e. MeMoLayer.retrieve
    without its final F.normalize.
    """

    def __init__(self, layer, store: BatchStore, n_holdout_batches: int = 1):
        if not getattr(layer, "layerized_CMM_OUT", False):
            raise ValueError("layer.CMM_OUT is required")
        if len(store) <= n_holdout_batches:
            raise ValueError("not enough batches: need at least n_holdout_batches + 1")
        m0 = store.manifest[0]
        if m0["keys_shape"][2:] != [layer.h, layer.d] or m0["vals_shape"][-1] != layer.d:
            raise ValueError(f"store shapes {m0['keys_shape']} / {m0['vals_shape']} "
                             f"do not match layer (h={layer.h}, d={layer.d})")

        self.layer, self.store = layer, store
        self.n_holdout = n_holdout_batches
        self.max_train_batches = len(store) - n_holdout_batches
        self.device = layer.CMM_OUT.weight.device
        self.dtype = layer.CMM_OUT.weight.dtype
        self.n_batches_stored = 0
        self.n_pairs_stored = 0
        self._cands: Optional[CandidateSet] = None
        if self.dtype != torch.float32:
            warnings.warn(f"CMM_OUT is {self.dtype}: accumulating many outer products in low "
                          "precision loses information regardless of capacity. Use float32.")

    @property
    def cands(self) -> CandidateSet:
        if self._cands is None:
            self._cands = CandidateSet(self.store)
        return self._cands

    # ---- memory management -----------------------------------------------
    @torch.no_grad()
    def reset(self):
        self.layer.CMM_OUT.reset_parameters()
        self.n_batches_stored = 0
        self.n_pairs_stored = 0

    @torch.no_grad()
    def memorize_batches(self, n_batches: int):
        """Replay batches [n_batches_stored, n_batches) exactly as they were originally given."""
        if n_batches > self.max_train_batches:
            raise ValueError(f"only {self.max_train_batches} batches available for memorization "
                             f"({self.n_holdout} held out)")
        if n_batches < self.n_batches_stored:
            raise ValueError("cannot un-memorize; call reset() first")
        for i in range(self.n_batches_stored, n_batches):
            k, v, ts, is_last = self.store.load(i, self.device, self.dtype)
            self.layer.memorize(k, v, to_save_sequences=ts, is_last=is_last)
            self.n_pairs_stored += self.store.n_pairs(i)
        self.n_batches_stored = n_batches

    def save_state(self, path):
        torch.save(dict(layer=self.layer.state_dict(), n_batches=self.n_batches_stored,
                        n_pairs=self.n_pairs_stored), path)

    def load_state(self, path):
        st = torch.load(path, map_location=self.device)
        self.layer.load_state_dict(st["layer"])
        self.n_batches_stored, self.n_pairs_stored = st["n_batches"], st["n_pairs"]

    # ---- encoding / recall ------------------------------------------------
    def _encode(self, k: torch.Tensor) -> torch.Tensor:
        B, L, H, D = k.shape
        seq_enc, per_token = self.layer.get_projections(k, L, H, D)
        return per_token if self.layer.compOp.name == "JLT" else seq_enc  # (B, L, d)

    def _pairs(self, i: int, with_values: bool = True):
        """All (encoded key, value) pairs of batch i selected by its to_save_sequences, flattened
        to (B * n_save, d). Flat index = b * n_save + j."""
        k = self.store.load_keys(i, self.device, self.dtype)
        enc = self._encode(k)
        ts = self.store.to_save(i)
        sel = list(ts) if ts is not None else None
        if sel is not None:
            enc = enc[:, sel]
        enc = enc.reshape(-1, enc.shape[-1])
        if not with_values:
            return enc, None
        v = self.store.load_values(i, self.device, self.dtype)
        if sel is not None:
            v = v[:, sel]
        return enc, v.reshape(-1, v.shape[-1])

    def _gather_holdout(self, max_n: int):
        encs, n = [], 0
        for i in range(self.max_train_batches, len(self.store)):
            e, _ = self._pairs(i, with_values=False)
            encs.append(e)
            n += e.shape[0]
            if n >= max_n:
                break
        return torch.cat(encs)[:max_n]

    def _top1(self, out: torch.Tensor, v_true: torch.Tensor) -> float:
        """Cleanup accuracy: is the true value (one of) the stored values nearest to the recall?
        Defined by similarity, not index, so duplicated values are handled correctly."""
        out_n = F.normalize(out, dim=-1)
        true_sim = (out_n * F.normalize(v_true, dim=-1)).sum(-1)
        best = torch.full_like(true_sim, -float("inf"))
        for i in range(self.n_batches_stored):
            _, bank = self._pairs_values_only(i)
            sims = out_n @ F.normalize(bank, dim=-1).T
            best = torch.maximum(best, sims.max(dim=1).values)
        return (true_sim >= best - 1e-4).float().mean().item()

    def _pairs_values_only(self, i: int):
        v = self.store.load_values(i, self.device, self.dtype)
        ts = self.store.to_save(i)
        if ts is not None:
            v = v[:, list(ts)]
        return None, v.reshape(-1, v.shape[-1])

    # ---- decoders ---------------------------------------------------------
    def _stored_counts(self) -> np.ndarray:
        ids = np.concatenate(self.cands.batch_ids[: self.n_batches_stored])
        return np.bincount(ids, minlength=self.cands.V)

    @torch.no_grad()
    def _fit_beta(self, out_c, S_true_c, lp_true_c, gamma, logprior, beta_th, n_beta, chunk):
        """Grid search (+-3 octaves around the theoretical beta) of the inverse temperature that
        minimises the NLL on the calibration rows, without and with the prior."""
        dev = out_c.device
        betas = beta_th * 2.0 ** torch.linspace(-3, 3, n_beta, device=dev)
        n = out_c.shape[0]
        lse_np = torch.full((n_beta, n), -float("inf"), device=dev)
        lse_p = torch.full((n_beta, n), -float("inf"), device=dev)
        for s, U, sq in self.cands.chunks(dev, out_c.dtype, chunk):
            S = out_c @ U.T - 0.5 * gamma * sq
            lp = logprior[s:s + U.shape[0]]
            for k in range(n_beta):
                Z = betas[k] * S
                lse_np[k] = torch.logaddexp(lse_np[k], torch.logsumexp(Z, dim=1))
                lse_p[k] = torch.logaddexp(lse_p[k], torch.logsumexp(Z + lp, dim=1))
        nll_np = (lse_np - betas[:, None] * S_true_c[None]).mean(1)
        nll_p = (lse_p - betas[:, None] * S_true_c[None] - lp_true_c[None]).mean(1)
        return betas[nll_np.argmin()].item(), betas[nll_p.argmin()].item()

    @torch.no_grad()
    def _eval_rows(self, out_e, gamma, logprior, beta_th, b_np, b_p, chunk, k):
        """One streaming pass over the candidates: log-normalisers and top-k for every decoder."""
        dev, n = out_e.device, out_e.shape[0]
        lse = {nm: torch.full((n,), -float("inf"), device=dev)
               for nm in ("dot", "th", "th_p", "fit", "fit_p")}
        tk = {nm: _TopK(n, k, dev) for nm in ("dot", "S", "th_p", "fit_p")}
        for s, U, sq in self.cands.chunks(dev, out_e.dtype, chunk):
            dot = out_e @ U.T
            S = dot - 0.5 * gamma * sq
            lp = logprior[s:s + U.shape[0]]
            Zth_p = beta_th * S + lp
            Zfit_p = b_p * S + lp
            lse["dot"] = torch.logaddexp(lse["dot"], torch.logsumexp(dot, 1))
            lse["th"] = torch.logaddexp(lse["th"], torch.logsumexp(beta_th * S, 1))
            lse["th_p"] = torch.logaddexp(lse["th_p"], torch.logsumexp(Zth_p, 1))
            lse["fit"] = torch.logaddexp(lse["fit"], torch.logsumexp(b_np * S, 1))
            lse["fit_p"] = torch.logaddexp(lse["fit_p"], torch.logsumexp(Zfit_p, 1))
            tk["dot"].update(dot, s)
            tk["S"].update(S, s)
            tk["th_p"].update(Zth_p, s)
            tk["fit_p"].update(Zfit_p, s)
        return lse, tk

    @torch.no_grad()
    def _decode(self, out, enc, v, true_ids, P, d, seed, prior_smoothing, chunk, n_beta,
                topk=DEFAULT_TOPK):
        dev = out.device
        n, V = out.shape[0], self.cands.V
        if n < 8:
            return None
        ks = _norm_ks(topk)
        k_max = min(max(ks), V)  # size of the streaming top-k buffer (never larger than V)

        # --- theoretical temperature: r = g*u_true + N(0, s2 I), s2 = ((P-1)/d) * sigma_v^2
        gamma = enc.pow(2).sum(-1).mean().item()
        sigma2 = v.pow(2).sum(-1).mean().item() / d
        s2 = max((P - 1) / d * sigma2, 1e-12)
        beta_th = gamma / s2

        # --- prior from what was actually memorized
        counts = torch.as_tensor(self._stored_counts(), device=dev, dtype=torch.float64)
        a = prior_smoothing
        logprior = torch.log((counts + a) / (counts.sum() + a * V)).to(out.dtype)

        # --- quantities of the true candidate
        U_true = self.cands.vecs[true_ids.cpu()].to(dev, out.dtype)
        dot_true = (out * U_true).sum(-1)
        S_true = dot_true - 0.5 * gamma * U_true.pow(2).sum(-1)
        lp_true = logprior[true_ids]

        # --- 2-fold cross-fitting: beta fitted on one half, decoders scored on the other half
        perm = torch.as_tensor(np.random.default_rng(seed + 7 + P).permutation(n), device=dev)
        folds = [perm[: n // 2], perm[n // 2:]]
        nll = {nm: torch.zeros(n, device=dev) for nm in DECODER_NAMES}
        hits = {nm: {k: torch.zeros(n, device=dev) for k in ks} for nm in DECODER_NAMES}
        fit_np, fit_p = [], []
        for f in (0, 1):
            cal, ev = folds[f], folds[1 - f]
            b_np, b_p = self._fit_beta(out[cal], S_true[cal], lp_true[cal], gamma, logprior,
                                       beta_th, n_beta, chunk)
            fit_np.append(b_np)
            fit_p.append(b_p)
            lse, tk = self._eval_rows(out[ev], gamma, logprior, beta_th, b_np, b_p, chunk, k_max)
            true_logit = {
                "dot_T1": dot_true[ev],
                "bayes_T": beta_th * S_true[ev],
                "bayes_T_prior": beta_th * S_true[ev] + lp_true[ev],
                "fit_T": b_np * S_true[ev],
                "fit_T_prior": b_p * S_true[ev] + lp_true[ev],
            }
            lse_of = {"dot_T1": lse["dot"], "bayes_T": lse["th"], "bayes_T_prior": lse["th_p"],
                      "fit_T": lse["fit"], "fit_T_prior": lse["fit_p"]}
            rank_of = {"dot_T1": tk["dot"], "bayes_T": tk["S"], "bayes_T_prior": tk["th_p"],
                       "fit_T": tk["S"], "fit_T_prior": tk["fit_p"]}
            t = true_ids[ev]
            for nm in DECODER_NAMES:
                nll[nm][ev] = lse_of[nm] - true_logit[nm]
                top = rank_of[nm].i  # (n_ev, k_max), sorted by decreasing score
                for k in ks:
                    kk = min(k, k_max)
                    hits[nm][k][ev] = (top[:, :kk] == t[:, None]).any(1).float()

        # --- reference baselines (prior only)
        top_prior = logprior.topk(k_max).indices
        prior_only = dict(nll=float(-lp_true.mean()))
        for k in ks:
            kk = min(k, k_max)
            prior_only[f"top{k}"] = float((true_ids[:, None] == top_prior[None, :kk]).any(1).float().mean())

        geo = lambda xs: float(math.exp(sum(math.log(x) for x in xs) / len(xs)))
        beta_fit, beta_fit_p = geo(fit_np), geo(fit_p)
        decs = {nm: dict(nll=float(nll[nm].mean()),
                         **{f"top{k}": float(hits[nm][k].mean()) for k in ks})
                for nm in DECODER_NAMES}
        return dict(
            topk=ks,
            n_candidates=int(V),
            n_distinct_stored=int((counts > 0).sum().item()),
            gamma=gamma,
            beta_theory=float(beta_th),
            beta_fit=beta_fit,
            beta_fit_prior=beta_fit_p,
            nll_uniform=float(math.log(V)),
            top1_chance=1.0 / V,
            prior_only=prior_only,
            info_from_memory_nats=prior_only["nll"] - decs["fit_T_prior"]["nll"],
            decoders=decs,
        )

    # ---- metrics ----------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, n_eval: int = 1000, n_fresh: int = 1000, seed: int = 0,
                 decode: bool = True, prior_smoothing: float = 1.0,
                 vocab_chunk: int = 8192, n_beta: int = 21,
                 topk: Sequence[int] = DEFAULT_TOPK) -> Dict:
        P, d, nb = self.n_pairs_stored, self.layer.d, self.n_batches_stored
        if nb == 0:
            raise ValueError("nothing memorized yet")

        # --- sample stored pairs uniformly over [0, P), group by batch to limit disk reads
        counts = np.array([self.store.n_pairs(i) for i in range(nb)])
        cum = np.cumsum(counts)
        rng = np.random.default_rng(seed + nb)
        sel = np.sort(rng.choice(P, size=min(n_eval, P), replace=False))
        b_of = np.searchsorted(cum, sel, side="right")
        local = sel - (cum[b_of] - counts[b_of])

        enc_l, val_l, id_l = [], [], []
        for b in np.unique(b_of):
            m = b_of == b
            enc_b, v_b = self._pairs(int(b))
            li = torch.as_tensor(local[m], device=self.device)
            enc_l.append(enc_b[li])
            val_l.append(v_b[li])
            if decode:
                id_l.append(self.cands.batch_ids[int(b)][local[m]])
        enc, v = torch.cat(enc_l), torch.cat(val_l)
        out = self.layer.CMM_OUT(enc)

        cos = F.cosine_similarity(out, v, dim=-1)
        rel = (out - v).norm(dim=-1) / v.norm(dim=-1)
        top1 = self._top1(out, v)

        # --- pure crosstalk from unseen (held-out) keys
        enc_f = self._gather_holdout(n_fresh)
        out_f = self.layer.CMM_OUT(enc_f)
        ratio = (out.pow(2).sum(-1).mean() / out_f.pow(2).sum(-1).mean()).item()

        W = self.layer.CMM_OUT.weight.float()
        alpha = P / d
        row = dict(
            n_batches=nb,
            P=P,
            alpha=alpha,
            cos_mean=cos.mean().item(),
            cos_p05=torch.quantile(cos.float(), 0.05).item(),
            cos_min=cos.min().item(),
            cos_theory=1.0 / math.sqrt(1.0 + alpha),
            rel_err=rel.mean().item(),
            rel_err_theory=math.sqrt(alpha),
            top1_acc=top1,
            top1_chance=1.0 / P,
            key_norm_mean=enc.norm(dim=-1).mean().item(),  # ~1 for non-JLT ops
            power_ratio=ratio,                    # ||M k_stored||^2 / ||M k_heldout||^2
            d_eff_est=P * (ratio - 1.0),          # ~ d for isotropic, uncorrelated keys/values
            fro2_per_pair=W.pow(2).sum().item() / (P * v.pow(2).sum(-1).mean().item()),
        )
        if decode:
            true_ids = torch.as_tensor(np.concatenate(id_l), device=self.device)
            row["dec"] = self._decode(out, enc, v, true_ids, P, d, seed,
                                      prior_smoothing, vocab_chunk, n_beta, topk=topk)
        return row

    def sweep(self, checkpoints: Sequence[int], n_eval: int = 1000, n_fresh: int = 1000,
              out_json: Optional[str] = None, reset: bool = True, verbose: bool = True,
              decode: bool = True, prior_smoothing: float = 1.0,
              topk: Sequence[int] = DEFAULT_TOPK):
        """checkpoints = numbers of batches to have memorized (see default_checkpoints).
        topk = list of k for which the top-k accuracy of every decoder is computed."""
        if reset:
            self.reset()
        rows = []
        for nb in sorted(checkpoints):
            self.memorize_batches(nb)
            r = self.evaluate(n_eval=n_eval, n_fresh=n_fresh, decode=decode,
                              prior_smoothing=prior_smoothing, topk=topk)
            rows.append(r)
            if verbose:
                print(f"batches={r['n_batches']:5d} P={r['P']:8d} alpha={r['alpha']:.3f}  "
                      f"cos={r['cos_mean']:.3f} (theory {r['cos_theory']:.3f}, p05 {r['cos_p05']:.3f})  "
                      f"top1={r['top1_acc']:.3f}  d_eff~{r['d_eff_est']:.0f}/{self.layer.d}  "
                      f"|k|={r['key_norm_mean']:.2f}")
                dd = r.get("dec")
                if dd:
                    D = dd["decoders"]
                    topk_str = " ".join(f"top{k}={D['fit_T_prior'][f'top{k}']:.3f}" for k in dd["topk"])
                    print(f"      NLL dot={D['dot_T1']['nll']:.2f} bayes={D['bayes_T']['nll']:.2f} "
                          f"bayes+prior={D['bayes_T_prior']['nll']:.2f} "
                          f"fit+prior={D['fit_T_prior']['nll']:.2f} "
                          f"(uniform {dd['nll_uniform']:.2f}, prior-only {dd['prior_only']['nll']:.2f})  "
                          f"fit+prior: {topk_str}  "
                          f"beta_fit/theory={dd['beta_fit_prior'] / dd['beta_theory']:.2f}  "
                          f"V={dd['n_candidates']}")
        if out_json:
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(out_json).write_text(json.dumps(rows, indent=2))
        return rows


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot_sweep(rows: List[Dict], path: Optional[str] = None):
    import matplotlib.pyplot as plt

    P = [r["P"] for r in rows]
    has_dec = all(r.get("dec") for r in rows)
    nrows = 2 if has_dec else 1
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 4 * nrows), squeeze=False)

    # ---- row 1: recall quality (unchanged) --------------------------------
    ax = axes[0]
    ax[0].plot(P, [r["cos_mean"] for r in rows], "o-", label="measured (mean)")
    ax[0].plot(P, [r["cos_p05"] for r in rows], "o:", label="measured (5th pct)")
    ax[0].plot(P, [r["cos_theory"] for r in rows], "--", label=r"theory $1/\sqrt{1+P/d}$")
    ax[0].set(xscale="log", xlabel="stored pairs P", ylabel="cosine(recall, target)")
    ax[0].legend()

    ax[1].plot(P, [r["top1_acc"] for r in rows], "o-", label="top-1 cleanup (cosine, stored bank)")
    ax[1].plot(P, [r["top1_chance"] for r in rows], "--", label="chance")
    ax[1].set(xscale="log", yscale="log", xlabel="stored pairs P", ylabel="accuracy")
    ax[1].legend()

    ax[2].plot(P, [r["d_eff_est"] for r in rows], "o-")
    ax[2].set(xscale="log", xlabel="stored pairs P", ylabel="d_eff estimate (held-out probe)")

    # ---- row 2: decoders ---------------------------------------------------
    if has_dec:
        a = axes[1]
        ks = rows[0]["dec"]["topk"]
        dec = lambda nm, key: [r["dec"]["decoders"][nm][key] for r in rows]
        label = {"dot_T1": "dot product, T=1 (current)",
                 "bayes_T": "Bayes temperature",
                 "bayes_T_prior": "Bayes temperature + prior",
                 "fit_T": "fitted temperature",
                 "fit_T_prior": "fitted temperature + prior"}
        sty = {"dot_T1": "x-", "bayes_T": "o--", "bayes_T_prior": "o-", "fit_T": "s--", "fit_T_prior": "s-"}

        # (a) negative log-likelihood
        for nm in DECODER_NAMES:
            a[0].plot(P, dec(nm, "nll"), sty[nm], label=label[nm])
        nll_u = [r["dec"]["nll_uniform"] for r in rows]
        nll_pr = [r["dec"]["prior_only"]["nll"] for r in rows]
        a[0].plot(P, nll_u, ":", color="gray", label="uniform (log V)")
        a[0].plot(P, nll_pr, "--", color="k", label="prior only")
        a[0].fill_between(P, dec("fit_T_prior", "nll"), nll_pr, color="C4", alpha=0.15,
                          label="information from memory")
        a[0].set(xscale="log", xlabel="stored pairs P", ylabel="NLL of the true value (nats)",
                 ylim=(0, 1.3 * max(max(nll_u), max(nll_pr))))
        a[0].legend(fontsize=7)

        # (b) accuracy of the decoders (no random baseline; linear y axis to resolve values near 1)
        a[1].plot(P, [r["top1_acc"] for r in rows], "k:", label="top-1 cosine cleanup (stored bank)")
        a[1].plot(P, dec("dot_T1", "top1"), "x-", label="top-1 dot product")
        a[1].plot(P, dec("bayes_T", "top1"), "o--", label="top-1 Bayes / fitted T (no prior)")
        a[1].plot(P, dec("bayes_T_prior", "top1"), "o-", label="top-1 Bayes T + prior")
        a[1].plot(P, dec("fit_T_prior", "top1"), "s-", label="top-1 fitted T + prior")
        for j, k in enumerate(k for k in ks if k > 1):
            a[1].plot(P, dec("fit_T_prior", f"top{k}"), "s--", color=f"C{5 + j}",
                      label=f"top-{k} fitted T + prior")
        a[1].set(xscale="log", xlabel="stored pairs P", ylabel="accuracy")
        a[1].set_ylim(top=1.01)  # lower limit is chosen automatically from the data
        a[1].legend(fontsize=7)

        # (c) inverse temperature: theory vs fitted
        a[2].plot(P, [r["dec"]["beta_theory"] for r in rows], "--", label=r"theory $\gamma/s^2$")
        a[2].plot(P, [r["dec"]["beta_fit"] for r in rows], "o-", label="fitted (no prior)")
        a[2].plot(P, [r["dec"]["beta_fit_prior"] for r in rows], "s-", label="fitted (with prior)")
        a[2].set(xscale="log", yscale="log", xlabel="stored pairs P",
                 ylabel=r"inverse temperature $\beta$")
        a[2].legend(fontsize=8)

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig