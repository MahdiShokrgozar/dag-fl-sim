# src/nodes/local_trainer.py
import copy
import logging
import math
import torch
from collections import defaultdict
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Above any sane CrossEntropy value for this domain (typically 0-10 even for a
# freshly-initialized model) - anything past this is treated as a diverged
# step, not a slow/bad-but-real one. See `train_and_evaluate`.
LOSS_EXPLOSION_CEILING = 1.0e6

# Well above a normally-trained CNN's typical weight range (2026-09-10,
# thesis-hub/docs/decisions.md same date - originally a >50 diagnostic-only
# threshold, made an active guard 2026-09-16). `_weights_are_finite` only
# catches NaN/Inf; an aggregate that is large but still finite (e.g. an
# extreme Non-IID node's locally-overfit classifier layer getting aggregated
# in) passes that check and can still blow up the very first forward pass,
# producing a loss explosion on EVERY batch of the round - the exact
# published-weights-equal-init_weights signature that false-flags an honest
# node as lazy/poisoning (thesis-hub/docs/decisions.md, 2026-09-16, P6).
# Fallback default only as of 2026-09-17 - overridable via
# local_training.init_weights_magnitude_ceiling; this value is what any run
# gets if that key is absent, so existing (MNIST) runs are unaffected.
INIT_WEIGHTS_MAGNITUDE_CEILING = 50.0

# Per-layer RMS (L2 norm / sqrt(numel)) ceiling (2026-09-16, thesis-hub/docs/
# decisions.md, P7's DIAG-EXPLOSION follow-up). INIT_WEIGHTS_MAGNITUDE_CEILING
# alone missed a real explosion (loss~7e8, output logits up to 1.6e9) because
# it was spread across fc1's wide 3136-dim input layer as many
# small-to-moderate values (every element under 50) rather than one huge
# element - fc1.weight measured RMS~10.0 in that run. A healthy trained
# layer's RMS is well under 1 in normal operation; 5.0 gives real margin
# above ordinary training variance while catching this kind of distributed
# explosion. Calibrated from MNIST only; confirmed 2026-09-17 (thesis-hub/
# docs/decisions.md same date) to be too tight for CIFAR-10, where a
# legitimate classifier-bias RMS of ~5-6 under Non-IID tripped this
# constantly and stalled training at chance accuracy with zero attackers
# present. Fallback default only as of 2026-09-17 - overridable via
# local_training.layer_rms_ceiling; this value is what any run gets if that
# key is absent, so existing (MNIST) runs are unaffected.
LAYER_RMS_CEILING = 5.0


def _weights_are_finite(weights: Dict[str, Any]) -> bool:
    """True iff every tensor in `weights` is entirely finite (no NaN/Inf)."""
    for v in weights.values():
        if torch.is_tensor(v) and not torch.isfinite(v).all():
            return False
    return True


def _weights_max_abs(weights: Dict[str, Any]) -> Tuple[float, str]:
    """(max |value| across all tensors, layer name it came from) - diagnostic
    only, for telling apart "already large-but-finite before this call even
    started" from "exploded during this call's own forward/backward pass"."""
    best_val, best_key = 0.0, ""
    for k, v in weights.items():
        if not torch.is_tensor(v) or v.numel() == 0:
            continue
        try:
            m = v.detach().abs().max().item()
        except Exception:
            continue
        if m > best_val:
            best_val, best_key = m, k
    return best_val, best_key


def _weights_max_rms(weights: Dict[str, Any]) -> Tuple[float, str]:
    """(max per-layer RMS = L2 norm / sqrt(numel), layer it came from).

    Confirmed necessary 2026-09-16 (thesis-hub/docs/decisions.md, P7's
    DIAG-EXPLOSION log): `_weights_max_abs` alone missed a real explosion
    because it was spread across a WIDE layer (this project's CNN has an
    fc1 with a 3136-dim input, `src/models/cnn.py`) as many small-to-moderate
    values rather than concentrated in one huge element - fc1.weight measured
    L2=6336 (RMS~10.0) with every individual element still under the
    per-element ceiling of 50. RMS (not raw L2) so the same threshold is
    meaningful across layers of very different sizes without hardcoding
    per-shape numbers.
    """
    best_val, best_key = 0.0, ""
    for k, v in weights.items():
        if not torch.is_tensor(v) or v.numel() == 0:
            continue
        try:
            rms = v.detach().float().norm().item() / (v.numel() ** 0.5)
        except Exception:
            continue
        if rms > best_val:
            best_val, best_key = rms, k
    return best_val, best_key


def _weights_top_layer_norms(weights: Dict[str, Any], top_n: int = 3) -> str:
    """Top-N layers by L2 norm (not max|value|) - a wide layer (e.g. a
    3136-input fc1 on this project's CNN) can drive an exploding dot product
    through many small-to-moderate weights summed over a large input
    dimension, which per-element max|value| (see `_weights_max_abs`) does not
    catch. Diagnostic for `train_and_evaluate`'s explosion logging - see
    thesis-hub/docs/decisions.md, 2026-09-16 (P7 showed the max|value|<50
    guard never fired on a round that still exploded to loss~7e8)."""
    norms = []
    for k, v in weights.items():
        if not torch.is_tensor(v) or v.numel() == 0:
            continue
        try:
            norms.append((v.detach().float().norm().item(), k, tuple(v.shape)))
        except Exception:
            continue
    norms.sort(reverse=True)
    return ", ".join(f"{k}(shape={shape}, l2={n:.4g})" for n, k, shape in norms[:top_n])


class LocalTrainer:
    """
    Handles local PyTorch model training and evaluation.
    Designed to be fully GENERIC and protocol-agnostic.
    """
    def __init__(self, node_ref: Any):
        self.node = node_ref

    def evaluate_state(self, model: torch.nn.Module, loader: Any = None) -> Tuple[float, Dict[int, float]]:
        """
        Independent evaluation method to establish baseline accuracy and class distribution.
        Args:
            model (torch.nn.Module): The PyTorch model to evaluate.
            loader: Optional data loader to evaluate against. Defaults to
                self.node.test_loader (this node's own local partition -
                see thesis-hub/docs/decisions.md, 2026-09-15). Callers pass
                an explicit loader to evaluate against something else - e.g.
                AccuracyMetric evaluates against node.simulator.global_test_loader
                for reporting purposes only, never for a protocol's own
                decisions.
        Returns:
            Tuple[float, Dict[int, float]]: Overall accuracy and per-class accuracy dictionary.
        """
        loader = loader if loader is not None else self.node.test_loader
        if not loader:
            logger.warning(
                f"[EVAL] Node {self.node.node_id} has no test_loader; "
                f"evaluate_state returning empty accuracy/class map."
            )
            return 0.0, {}

        model.eval()

        # Per-class counts stay on the device and are read back once per call.
        # The earlier per-sample loop called .item() twice per test image, i.e.
        # one GPU sync each - ~7.5M syncs in a 600s run, for identical counts
        # (thesis-hub/docs/decisions.md, 2026-09-28).
        # first_seen keeps the dict in first-appearance order, as the loop did:
        # per_class_acc is serialized into block metadata, so key order matters.
        class_correct = class_total = first_seen = None
        offset = 0

        with torch.inference_mode():
            for data, target in loader:
                data, target = data.to(self.node.device), target.to(self.node.device)
                outputs = model(data)
                predicted = outputs.argmax(dim=1)

                target = target.long()
                if target.numel() == 0:
                    continue
                n = int(target.max()) + 1
                if class_total is None:
                    class_total = torch.zeros(n, dtype=torch.long, device=target.device)
                    class_correct = torch.zeros(n, dtype=torch.long, device=target.device)
                    first_seen = torch.full((n,), torch.iinfo(torch.long).max, dtype=torch.long, device=target.device)
                elif n > class_total.numel():
                    pad = n - class_total.numel()
                    class_total = torch.nn.functional.pad(class_total, (0, pad))
                    class_correct = torch.nn.functional.pad(class_correct, (0, pad))
                    first_seen = torch.cat([first_seen, torch.full((pad,), torch.iinfo(torch.long).max,
                                                                   dtype=torch.long, device=target.device)])
                size = class_total.numel()
                class_total += torch.bincount(target, minlength=size)
                class_correct += torch.bincount(target[predicted == target], minlength=size)
                positions = torch.arange(offset, offset + target.numel(), device=target.device)
                first_seen.scatter_reduce_(0, target, positions, reduce="amin")
                offset += target.numel()

        if class_total is None:
            totals, corrects, order = [], [], []
        else:
            totals, corrects = class_total.tolist(), class_correct.tolist()
            seen = first_seen.tolist()
            order = sorted((k for k in range(len(totals)) if totals[k] > 0), key=lambda k: seen[k])
        total_samples = sum(totals)
        accuracy = 100.0 * sum(corrects) / total_samples if total_samples > 0 else 0.0

        per_class_acc = {k: corrects[k] / totals[k] * 100.0 for k in order}
        
        if not per_class_acc:
            logger.warning(
                f"[EVAL] Node {self.node.node_id} test_loader yielded zero samples; "
                f"per_class_acc is empty."
            )
        
        return accuracy, per_class_acc

    def train_and_evaluate(self, init_weights: Dict[str, Any]) -> Tuple[Dict[str, Any], float, float]:
        active_model = None

        # --- 0. Reject an already-corrupted aggregate before training on it ---
        # Confirmed 2026-09-10 (thesis-hub/docs/decisions.md same date): a
        # diagnostic Kaggle run showed ~23% of training calls starting with an
        # already-astronomical loss (up to ~8e8) on a perfectly normal batch
        # (data/labels both sane per DIAG-BATCH0) - the corruption was already
        # in `init_weights` before this call did anything, and it then spread
        # further via aggregation (a neighboring node exploded within the same
        # second). Training further on top of a non-finite/exploded aggregate
        # only deepens and spreads the corruption. Falls back to this node's
        # own last known-good weights instead of blindly adopting the bad one -
        # loses this round's aggregation benefit for this node, but that is
        # far better than contributing another exploded model back into the
        # network.
        if not _weights_are_finite(init_weights):
            logger.error(
                f"🚨 [DIVERGENCE-GUARD] Node {self.node.node_id}: incoming aggregated "
                f"weights contain NaN/Inf - refusing to train on them, falling back "
                f"to this node's own last known-good weights."
            )
            init_weights = getattr(self.node, "weights", init_weights)
        else:
            # Made an active guard 2026-09-16 (thesis-hub/docs/decisions.md,
            # same date - was diagnostic-only since 2026-09-10). Confirmed via
            # a real Kaggle run (P6) that a large-but-finite init_weights
            # (passes `_weights_are_finite`) can still blow up the very first
            # forward pass on EVERY batch of the round - every batch discards,
            # the model never takes a single optimizer step, and the published
            # weights end up bit-for-bit equal to init_weights: the exact
            # exact-zero signature the S1 lazy/poison gates read as malicious.
            # weight_decay (below) slows the walk toward this state but does
            # not prevent inheriting an already-extreme aggregate from a peer.
            # Same treatment as the NaN/Inf branch above and for the same
            # reason: training further on an unsafe base only risks
            # republishing it (or something derived from it) as this node's
            # own contribution - falling back to this node's own last
            # known-good weights loses this round's aggregation benefit but
            # keeps this node out of the network's contamination path.
            max_abs, max_key = _weights_max_abs(init_weights)
            # RMS check added 2026-09-16 (thesis-hub/docs/decisions.md, P7's
            # DIAG-EXPLOSION follow-up): max_abs alone missed a real explosion
            # (loss~7e8) that was spread across a wide layer (fc1, 3136-dim
            # input) as many small-to-moderate values rather than one huge
            # element - every element stayed under INIT_WEIGHTS_MAGNITUDE_CEILING
            # while fc1.weight's RMS still measured ~10.0. See LAYER_RMS_CEILING.
            max_rms, rms_key = _weights_max_rms(init_weights)
            # Both ceilings were calibrated from one MNIST incident and are
            # architecture/dataset-specific, not universal (2026-09-17,
            # thesis-hub/docs/decisions.md same date - CIFAR-10 honest-only
            # pilot: legitimate fc2.bias RMS ~5-6 under Non-IID tripped the
            # MNIST-tuned default constantly, stalling training at chance
            # accuracy with zero attackers present). Config-overridable per
            # run; fallback matches the original hardcoded values exactly, so
            # any run that doesn't set these keys behaves bit-for-bit as
            # before.
            local_training_cfg = self.node.config.get("local_training", {})
            init_weights_magnitude_ceiling = local_training_cfg.get(
                "init_weights_magnitude_ceiling", INIT_WEIGHTS_MAGNITUDE_CEILING
            )
            layer_rms_ceiling = local_training_cfg.get(
                "layer_rms_ceiling", LAYER_RMS_CEILING
            )
            if max_abs > init_weights_magnitude_ceiling or max_rms > layer_rms_ceiling:
                logger.error(
                    f"🚨 [DIVERGENCE-GUARD] Node {self.node.node_id}: incoming aggregated "
                    f"weights are finite but already extreme (max|value|={max_abs:.4g} in "
                    f"layer '{max_key}', max layer RMS={max_rms:.4g} in layer '{rms_key}') - "
                    f"refusing to train on them, falling back to this node's own last "
                    f"known-good weights."
                )
                init_weights = getattr(self.node, "weights", init_weights)

        # --- 1. Model Checkout ---
        try:
            active_model = self.node.cache_manager.checkout_model(init_weights)
        except Exception as e:
            logger.warning(f"Model cache checkout failed ({e}). Using template fallback.")
            try:
                active_model = copy.deepcopy(self.node.cache_manager.get_template())
                active_model.load_state_dict(init_weights)
                active_model.to(self.node.device)
            except Exception as fallback_e:
                self.node.logger.error(f"Fallback model instantiation failed: {fallback_e}")
                return init_weights, 0.0, 0.0
            
        # Calculate simulated compute time
        base_time = self.node.config.get("simulation", {}).get("base_training_time_per_epoch", 1.0)
        straggler_penalty = getattr(self.node, "delay_multiplier", 1.0)
        compute_time = (base_time * self.node.epochs * straggler_penalty) / self.node.capacity
        
        # --- 2. Dynamic Hyper-parameters Configuration ---
        train_cfg = self.node.config.get("local_training", {})
        lr = train_cfg.get("lr", 0.01)
        momentum = train_cfg.get("momentum", 0.0)
        opt_name = train_cfg.get("optimizer", "sgd").lower()
        max_batches = train_cfg.get("max_batches", 0)
        clip_grad = train_cfg.get("clip_grad_norm", 0.5)
        
        active_model.train()
        
        # weight_decay (2026-09-10, thesis-hub/docs/decisions.md same date):
        # root-cause fix for the divergence issue below. DIAG-INIT-MAGNITUDE
        # confirmed a proposer's own classifier-layer weight (fc1.weight) can
        # drift to magnitude ~50+ over a run's ~100 training rounds with no
        # regularization pulling it back - plausibly driven by severe Non-IID
        # (dirichlet_alpha=0.1, near-single-class shards for some nodes)
        # combined with a fairly aggressive lr/momentum. clip_grad_norm_ below
        # only bounds the gradient PER STEP; it cannot undo cumulative growth
        # across many accepted steps. Standard L2 regularization, untuned
        # (5e-4 is a common default for small CNNs with SGD) - if seed-43-style
        # divergence persists, this value needs revisiting with real evidence,
        # not another guess.
        weight_decay = train_cfg.get("weight_decay", 5e-4)
        if opt_name == "adam":
            optimizer = torch.optim.Adam(active_model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.SGD(active_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            
        criterion = torch.nn.CrossEntropyLoss()

        # Diagnostic only (2026-09-10, thesis-hub/docs/decisions.md same date):
        # investigating the gpu_cache/cache_on_device regression (mixed_unified
        # collapsed to ~15% max accuracy, honest nodes stuck near random-guess,
        # when it was briefly enabled). Logs once per training call so it's
        # cheap; remove once the mechanism is confirmed and a real fix lands.
        _diag_first_loss = None
        _diag_last_loss = None
        _diag_logged_batch0 = False
        _diag_logged_explosion = False

        # --- 3. Local Training Loop ---
        diverged = False
        diverged_batches = 0
        for epoch in range(self.node.epochs):
            count = 0
            attempts = 0
            if self.node.train_loader:
                for data, target in self.node.train_loader:
                    # BUG (2026-09-10, found the same day via a seed-43 re-run):
                    # this used to gate on `count` (successful steps only). Once
                    # a batch could be skipped without stopping the round
                    # (the "continue" change below), a round where EVERY batch
                    # diverges never reaches `count >= max_batches` - `count`
                    # never moves - so the loop silently consumed the entire
                    # data loader (282 attempts observed on one node, versus the
                    # intended budget of 20) instead of stopping. Gate on total
                    # ATTEMPTS (successful + discarded) so a fully-diverging
                    # round still costs at most max_batches tries, matching the
                    # original budget, while a partially-diverging round still
                    # gets its full max_batches worth of chances to find a good
                    # batch.
                    if max_batches > 0 and attempts >= max_batches:
                        break
                    attempts += 1

                    data, target = data.to(self.node.device), target.long()

                    if not _diag_logged_batch0:
                        _diag_logged_batch0 = True
                        logger.debug(
                            f"[DIAG-BATCH0] node={self.node.node_id} "
                            f"is_on_gpu={getattr(self.node.train_loader, 'is_on_gpu', 'n/a')} "
                            f"data_shape={tuple(data.shape)} data_dtype={data.dtype} "
                            f"data_device={data.device} data_min={data.min().item():.4f} "
                            f"data_max={data.max().item():.4f} "
                            f"target_dtype={target.dtype} target_device={target.device} "
                            f"target_unique={sorted(set(target.detach().cpu().tolist()))} "
                            f"target_min={target.min().item()} target_max={target.max().item()}"
                        )

                    optimizer.zero_grad()

                    output = active_model(data)
                    loss = criterion(output, target)
                    loss.backward()

                    loss_val = loss.item()
                    if _diag_first_loss is None:
                        _diag_first_loss = loss_val
                    _diag_last_loss = loss_val

                    # Divergence guard (2026-09-10, thesis-hub/docs/decisions.md
                    # same date): confirmed via direct log evidence that ~23% of
                    # training calls in a diagnostic run diverged to loss values
                    # up to ~8e8 on perfectly normal input data - a genuine
                    # numerical explosion, not a data/label bug. clip_grad_norm_
                    # bounds the gradient AFTER backward, but does nothing once a
                    # step has already produced non-finite/exploded weights - by
                    # then the damage is done and, worse, it gets returned as
                    # this node's contribution and aggregated into others.
                    # Skip the update instead of applying it: keep the model at
                    # its last finite state.
                    #
                    # Skips only THIS batch and continues to the rest (2026-09-10,
                    # same day, second pass) - aborting the whole round here used
                    # to guarantee the published weights equal init_weights
                    # EXACTLY (checkout_model() already loaded init_weights and no
                    # step ever applied), which is bit-for-bit the S1 signal's
                    # exact-zero "verbatim republish" lazy signature. Confirmed via
                    # DIAG-DIVERGED-OUTPUT: 452/452 discarded rounds landed at
                    # relative distance exactly 0.0 - a real Kaggle run then showed
                    # this cascading into false lazy/steal quarantines against 7 of
                    # 8 honest nodes in one cell. Continuing lets the remaining
                    # batches (max_batches, typically 20) still produce a genuine,
                    # nonzero update if they don't ALSO diverge - no worse than
                    # today's behavior in the worst case (every batch diverges =
                    # same all-batches-skipped outcome as before), strictly better
                    # otherwise. Deliberately not a self-reported "skip me" flag on
                    # the block - the project's evidence-verification design never
                    # trusts a self-report, and a genuinely lazy node could always
                    # claim it.
                    if not math.isfinite(loss_val) or loss_val > LOSS_EXPLOSION_CEILING:
                        # Diagnostic (2026-09-16, thesis-hub/docs/decisions.md same
                        # date): P7 showed INIT_WEIGHTS_MAGNITUDE_CEILING (per-element
                        # max|value| > 50) never fires on a round that still explodes
                        # to loss~7e8 - so per-element magnitude is not the (or not
                        # the only) driver. Logged once per round, at ERROR so it
                        # actually lands in simulation.log (the 2026-09-10 DIAG-*
                        # helpers are debug-level and were invisible in that run).
                        # Captures what max|value| cannot: a wide layer (fc1 here has
                        # a 3136-dim input) can drive an exploding dot product through
                        # many small-to-moderate weights summed over a large input
                        # dimension - top_layer_norms surfaces that by L2 norm, not
                        # per-element max. Also captures input data range and the
                        # model's own output logits range, to rule out a bad batch
                        # (already suspected NOT to be it per the 2026-09-10 note
                        # above, but re-confirmed here rather than assumed) versus a
                        # bad model state.
                        if not _diag_logged_explosion:
                            _diag_logged_explosion = True
                            try:
                                out_min = output.detach().min().item()
                                out_max = output.detach().max().item()
                                out_absmax = output.detach().abs().max().item()
                            except Exception:
                                out_min = out_max = out_absmax = float("nan")
                            top_layers = _weights_top_layer_norms(active_model.state_dict())
                            logger.error(
                                f"🔬 [DIAG-EXPLOSION] Node {self.node.node_id}: loss={loss_val:.4g} - "
                                f"data_min={data.min().item():.4g} data_max={data.max().item():.4g} "
                                f"data_finite={bool(torch.isfinite(data).all())} | "
                                f"output_min={out_min:.4g} output_max={out_max:.4g} "
                                f"output_absmax={out_absmax:.4g} | top_layers_by_l2={top_layers}"
                            )
                        logger.error(
                            f"🚨 [DIVERGENCE-GUARD] Node {self.node.node_id}: loss={loss_val:.4g} "
                            f"at batch {count} - discarding this step (not applying the update), "
                            f"continuing to the remaining batches this round."
                        )
                        optimizer.zero_grad()
                        diverged = True
                        diverged_batches += 1
                        continue

                    if clip_grad > 0.0:
                        torch.nn.utils.clip_grad_norm_(active_model.parameters(), max_norm=clip_grad)

                    optimizer.step()
                    count += 1

        if _diag_first_loss is not None:
            logger.debug(
                f"[DIAG-LOSS] node={self.node.node_id} "
                f"first_loss={_diag_first_loss:.4f} last_loss={_diag_last_loss:.4f} "
                f"delta={_diag_first_loss - _diag_last_loss:.4f} batches={count} "
                f"diverged_batches={diverged_batches}"
            )

        # --- 4. Local Evaluation Phase ---
        accuracy, per_class_acc = self.evaluate_state(active_model)
        
        # State Update
        self.node.per_class_accuracy = per_class_acc
        self.node.node_accuracy = accuracy

        if accuracy <= 0.0 and getattr(self.node, 'last_accuracy', 0.0) > 0.0:
            accuracy = self.node.last_accuracy
            self.node.node_accuracy = accuracy

        # --- 5. Clean up and Return ---
        new_weights = {k: v.cpu().clone() for k, v in active_model.state_dict().items()}

        # Diagnostic only (2026-09-10, thesis-hub/docs/decisions.md same
        # date, second pass): originally confirmed 452/452 fully-discarded
        # rounds landed at EXACTLY relative distance 0 from init_weights -
        # the S1 lazy gate's exact signature, and this cascaded into false
        # lazy/steal quarantines against 7 of 8 honest nodes in one seed-43
        # cell. The batch loop above now only skips the diverging batch and
        # keeps training on the rest, so this should mostly show a nonzero
        # distance going forward (weight_decay should also make
        # `diverged_batches` rarer in the first place) - kept to confirm that
        # empirically rather than assume it.
        if diverged:
            try:
                num = 0.0
                den = 0.0
                for k in new_weights:
                    if k in init_weights and torch.is_tensor(init_weights[k]):
                        num += (new_weights[k].cpu() - init_weights[k].cpu()).float().norm().item() ** 2
                        den += init_weights[k].cpu().float().norm().item() ** 2
                rel_dist = (num ** 0.5) / ((den ** 0.5) + 1e-9)
                logger.warning(
                    f"🔬 [DIAG-DIVERGED-OUTPUT] Node {self.node.node_id}: "
                    f"{diverged_batches} of {count + diverged_batches} batches were "
                    f"discarded this round - published weights are at relative "
                    f"distance {rel_dist:.6g} from init_weights (near 0 would look "
                    f"exactly like the lazy/steal signature to a validator)."
                )
            except Exception as e:
                logger.warning(f"[DIAG-DIVERGED-OUTPUT] comparison failed: {e}")

        try:
            self.node.cache_manager.checkin_model(active_model)
        except Exception:
            pass
            
        return new_weights, compute_time, accuracy