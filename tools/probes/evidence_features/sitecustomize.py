"""
Evidence-feature probe (experiment E0, thesis-hub/papers/drafts/architecture-paper-outline.md,
section 3.7; decisions.md 2026-09-30).

For every deep inspection a validator performs, it records the candidate features of the
proposed two-tier evidence, so their separation between honest and attacking proposers can
be measured before any consensus code changes:

  T1 (weights only)  d_declared   L2 distance of the block from the base its proposer declared
                                  (its training displacement; ~0 for a lazy republish)
                     d_detect     L2 distance from the validator's median detection base
                     nn_other_*   nearest earlier block by ANOTHER author in a recent window:
                                  distance, author, whether that block is among this block's
                                  aggregation refs (a noisy thief copies a block it did not
                                  declare)
  T2 (validator data) acc_block / acc_base and the 10x10 confusion matrices of the block and
                     of its declared base on the validator's own local test split (label
                     flipping moves predictions of the source class to one target class)

Output: <run_dir>/evidence_features.csv at process exit. Enabled per plan with `probe:
evidence_features` (run_experiments.py puts this folder first on PYTHONPATH). No simulator
code is changed. The probe draws no random numbers: the local test loader does not shuffle
and models are evaluated in eval mode.

Environment (optional): FEATURES_WINDOW (recent blocks kept for nn_other, default 80),
FEATURES_MAX_VAL (validation samples per evaluation, default 1000), FEATURES_EVAL_EVERY
(evaluate on data every k-th inspection, default 1; the T1 columns are always written).
"""

import atexit
import csv
import os
import sys
from collections import OrderedDict

_ROWS = []
_WINDOW = OrderedDict()  # block hash -> (author, timestamp, flat float32 tensor)
_WMAX = int(os.environ.get("FEATURES_WINDOW", "80"))
_MAXVAL = int(os.environ.get("FEATURES_MAX_VAL", "1000"))
_EVERY = max(1, int(os.environ.get("FEATURES_EVAL_EVERY", "1")))
_COUNT = [0]
_COLS = ["validator", "proposer", "block_hash", "block_ts", "sim_time", "n_refs", "declared",
         "d_declared", "d_detect", "nn_other_d", "nn_other_author", "nn_other_in_refs",
         "nn_other_ts_gap", "n_val", "acc_block", "acc_base", "conf_block", "conf_base"]


def _flat(weights):
    import torch
    parts = [v.detach().float().reshape(-1).cpu() for k, v in sorted(weights.items())
             if isinstance(v, torch.Tensor) and v.is_floating_point()]
    return torch.cat(parts) if parts else None


def _confusion(node, weights):
    """10x10 counts (row = true class) and n on the node's local test split, or None."""
    import torch
    loader = getattr(node, "test_loader", None)
    cm = getattr(node, "cache_manager", None)
    if loader is None or cm is None or not weights:
        return None
    model = cm.checkout_model(weights)
    try:
        model.eval()
        counts = torch.zeros(10, 10, dtype=torch.long)
        seen = 0
        with torch.no_grad():
            for data, target in loader:
                if seen >= _MAXVAL:
                    break
                take = min(len(target), _MAXVAL - seen)
                data, target = data[:take], target[:take]
                pred = model(data.to(getattr(node, "device", "cpu"))).argmax(dim=1).cpu()
                t = target.cpu().long()
                counts.index_put_((t, pred), torch.ones_like(t), accumulate=True)
                seen += take
        return counts, seen
    finally:
        cm.checkin_model(model)


def _row(consensus, frame_locals, declared_base):
    import torch
    proposer = frame_locals.get("proposer_id")
    block = frame_locals.get("block") or {}
    w_new = frame_locals.get("reconstructed_weights")
    detect = frame_locals.get("detection_base")
    refs = list(frame_locals.get("aggregation_refs") or [])
    if not w_new or proposer is None:
        return
    h = block.get("hash", block.get("id", ""))
    ts = float(block.get("timestamp", 0.0) or 0.0)
    node = getattr(consensus, "node", None)
    f_new = _flat(w_new)
    base = declared_base if declared_base else detect
    f_base = _flat(base) if base else None
    f_det = _flat(detect) if detect else None
    d_decl = float(torch.norm(f_new - f_base)) if f_base is not None and f_base.numel() == f_new.numel() else None
    d_det = float(torch.norm(f_new - f_det)) if f_det is not None and f_det.numel() == f_new.numel() else None
    best = (None, None, None, None)
    for bh, (author, bts, fv) in _WINDOW.items():
        if author == proposer or bts >= ts or fv.numel() != f_new.numel():
            continue
        d = float(torch.norm(f_new - fv))
        if best[0] is None or d < best[0]:
            best = (d, author, bh in refs, ts - bts)
    _COUNT[0] += 1
    conf_b = conf_p = None
    if node is not None and _COUNT[0] % _EVERY == 0:
        conf_b = _confusion(node, w_new)
        conf_p = _confusion(node, base) if base else None
    sim = getattr(getattr(node, "simulator", None), "current_time", None) if node is not None else None
    acc = lambda c: (float(c[0].diag().sum()) / c[1]) if c and c[1] else None
    flat_cm = lambda c: ",".join(str(int(x)) for x in c[0].reshape(-1)) if c else ""
    _ROWS.append([getattr(node, "node_id", ""), proposer, h, ts, sim, len(refs), bool(declared_base),
                  d_decl, d_det, best[0], best[1], best[2], best[3],
                  conf_b[1] if conf_b else None, acc(conf_b), acc(conf_p), flat_cm(conf_b), flat_cm(conf_p)])
    if h and h not in _WINDOW:
        _WINDOW[h] = (proposer, ts, f_new)
        while len(_WINDOW) > _WMAX:
            _WINDOW.popitem(last=False)


def _install():
    try:
        from src.strategies.consensus import farabi_consensus as fc
    except Exception:
        return
    orig = fc.FarabiConsensus._declared_base

    def _declared_base(self, coefficients, aggregation_refs):
        out = orig(self, coefficients, aggregation_refs)
        caller = sys._getframe(1)
        if caller.f_code.co_name == "_validate_deep_inspection":
            try:
                _row(self, caller.f_locals, out)
            except Exception as e:  # a probe must never break a run
                _ROWS.append(["<probe error>", repr(e)] + [None] * (len(_COLS) - 2))
        return out

    fc.FarabiConsensus._declared_base = _declared_base


def _dump():
    argv = list(sys.argv)
    if "--config" not in argv or not _ROWS:
        return
    d = os.path.dirname(os.path.abspath(argv[argv.index("--config") + 1]))
    with open(os.path.join(d, "evidence_features.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_COLS)
        w.writerows(_ROWS)


_install()
atexit.register(_dump)
