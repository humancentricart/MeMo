"""
Saturation evaluation for MeMoLayer correlation matrix memories, using the
REAL keys/values produced upstream (input_sequence / output_symbols).

Disk layout (one folder per layer, one file pair per memorize() call)
---------------------------------------------------------------------
    root/layer_{l}/batch_000000.keys.pt   input_sequence  (B, L, h, d)   shapes untouched
    root/layer_{l}/batch_000000.vals.pt   output_symbols  (B, L, d)
    root/layer_{l}/manifest.jsonl         one line per batch: shapes, to_save_sequences, n_pairs, is_last

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
                    n_eval=2000, n_fresh=2000, out_json="memo_dump/layer2_sweep.json")
    plot_sweep(rows, "memo_dump/layer2_sweep.png")

A "pair" is one (key, value) at a position selected by to_save_sequences, so one
batch holds B * len(to_save_sequences) pairs. The last `n_holdout_batches` batches are
never memorized; they provide unseen keys to measure pure crosstalk.
"""
from __future__ import annotations

import functools
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


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
        if self.dtype != torch.float32:
            warnings.warn(f"CMM_OUT is {self.dtype}: accumulating many outer products in low "
                          "precision loses information regardless of capacity. Use float32.")

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

    # ---- metrics ----------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, n_eval: int = 1000, n_fresh: int = 1000, seed: int = 0) -> Dict[str, float]:
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

        enc_l, val_l = [], []
        for b in np.unique(b_of):
            enc, v = self._pairs(int(b))
            li = torch.as_tensor(local[b_of == b], device=self.device)
            enc_l.append(enc[li])
            val_l.append(v[li])
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
        return dict(
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

    def sweep(self, checkpoints: Sequence[int], n_eval: int = 1000, n_fresh: int = 1000,
              out_json: Optional[str] = None, reset: bool = True, verbose: bool = True):
        """checkpoints = numbers of batches to have memorized (see default_checkpoints)."""
        if reset:
            self.reset()
        rows = []
        for nb in sorted(checkpoints):
            self.memorize_batches(nb)
            r = self.evaluate(n_eval=n_eval, n_fresh=n_fresh)
            rows.append(r)
            if verbose:
                print(f"batches={r['n_batches']:5d} P={r['P']:8d} alpha={r['alpha']:.3f}  "
                      f"cos={r['cos_mean']:.3f} (theory {r['cos_theory']:.3f}, p05 {r['cos_p05']:.3f})  "
                      f"top1={r['top1_acc']:.3f}  d_eff~{r['d_eff_est']:.0f}/{self.layer.d}  "
                      f"|k|={r['key_norm_mean']:.2f}")
        if out_json:
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(out_json).write_text(json.dumps(rows, indent=2))
        return rows


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot_sweep(rows: List[Dict[str, float]], path: Optional[str] = None):
    import matplotlib.pyplot as plt

    P = [r["P"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].plot(P, [r["cos_mean"] for r in rows], "o-", label="measured (mean)")
    ax[0].plot(P, [r["cos_p05"] for r in rows], "o:", label="measured (5th pct)")
    ax[0].plot(P, [r["cos_theory"] for r in rows], "--", label=r"theory $1/\sqrt{1+P/d}$")
    ax[0].set(xscale="log", xlabel="stored pairs P", ylabel="cosine(recall, target)")
    ax[0].legend()

    ax[1].plot(P, [r["top1_acc"] for r in rows], "o-", label="top-1 cleanup")
    ax[1].plot(P, [r["top1_chance"] for r in rows], "--", label="chance")
    ax[1].set(xscale="log", yscale="log", xlabel="stored pairs P", ylabel="accuracy")
    ax[1].legend()

    ax[2].plot(P, [r["d_eff_est"] for r in rows], "o-")
    ax[2].set(xscale="log", xlabel="stored pairs P", ylabel="d_eff estimate (held-out probe)")

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig