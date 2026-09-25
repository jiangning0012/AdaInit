# -*- coding: utf-8 -*-
"""State-only building blocks used by AdaInit.

The classes in this module deliberately do not depend on a model or a dataset.
Keeping detection and cache management separate from the adaptation loop makes
the corresponding paper ablations cheap to configure and straightforward to
unit test.
"""

import copy
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import torch


def clone_state_to_cpu(value: Any) -> Any:
    """Deep-clone nested optimizer/runtime state without retaining GPU storage."""

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_state_to_cpu(item) for item in value)
    return copy.deepcopy(value)


@dataclass
class DomainSignature:
    """Diagonal-Gaussian summary of shallow spatial/token features."""

    mean: torch.Tensor
    variance: torch.Tensor

    def clone_cpu(self) -> "DomainSignature":
        return DomainSignature(
            mean=self.mean.detach().cpu().clone(),
            variance=self.variance.detach().cpu().clone(),
        )

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"mean": self.mean, "variance": self.variance}

    @classmethod
    def from_state_dict(cls, state: Dict[str, torch.Tensor]) -> "DomainSignature":
        return cls(mean=state["mean"], variance=state["variance"])


@dataclass
class KnowledgeFingerprint:
    """Recent adaptable-parameter response used to describe learned knowledge.

    The payload remains one hard adaptable state.  This fingerprint records the
    parameter displacement produced by a fixed-length causal singleton window;
    it is metadata for cache diversity and retrieval, never a parameter mixture.
    """

    updates: Dict[str, torch.Tensor]

    def clone_cpu(self) -> "KnowledgeFingerprint":
        return KnowledgeFingerprint(
            updates={
                name: value.detach().cpu().clone()
                for name, value in self.updates.items()
            }
        )

    def state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {"updates": self.updates}

    @classmethod
    def from_state_dict(
        cls, state: Dict[str, Dict[str, torch.Tensor]]
    ) -> "KnowledgeFingerprint":
        return cls(updates=state["updates"]).clone_cpu()


def build_knowledge_fingerprint(
    start_state: Dict[str, torch.Tensor],
    end_state: Dict[str, torch.Tensor],
) -> KnowledgeFingerprint:
    """Build a local adaptation vector in the shared adaptable-state space."""

    if set(start_state) != set(end_state):
        raise ValueError("Knowledge-fingerprint state keys do not match.")
    return KnowledgeFingerprint(
        updates={
            name: (
                end_state[name].detach().cpu().float()
                - start_state[name].detach().cpu().float()
            )
            for name in sorted(start_state)
        }
    )


def knowledge_fingerprint_statistics(
    fingerprint: KnowledgeFingerprint,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Return lightweight norm diagnostics without flattening all layers."""

    squared_norm = 0.0
    nonzero_layers = 0
    for update in fingerprint.updates.values():
        norm = float(torch.linalg.vector_norm(update.float()).item())
        squared_norm += norm * norm
        if norm > eps:
            nonzero_layers += 1
    return {
        "total_norm": squared_norm**0.5,
        "nonzero_layers": float(nonzero_layers),
    }


def knowledge_fingerprint_similarity(
    lhs: KnowledgeFingerprint,
    rhs: KnowledgeFingerprint,
    eps: float = 1e-6,
) -> float:
    """ZOA-style layer-balanced cosine similarity of local update vectors."""

    if set(lhs.updates) != set(rhs.updates):
        raise ValueError("Knowledge-fingerprint keys do not match.")
    similarities = []
    for name in sorted(lhs.updates):
        lhs_update = lhs.updates[name].float().reshape(-1)
        rhs_update = rhs.updates[name].float().reshape(-1)
        lhs_norm = torch.linalg.vector_norm(lhs_update)
        rhs_norm = torch.linalg.vector_norm(rhs_update)
        if lhs_norm <= eps and rhs_norm <= eps:
            similarity = 1.0
        elif lhs_norm <= eps or rhs_norm <= eps:
            similarity = 0.0
        else:
            similarity = float(
                torch.dot(lhs_update, rhs_update).div(lhs_norm * rhs_norm).item()
            )
        similarities.append(similarity)
    if not similarities:
        return 0.0
    return float(sum(similarities) / len(similarities))


def knowledge_fingerprint_distance(
    lhs: KnowledgeFingerprint,
    rhs: KnowledgeFingerprint,
    eps: float = 1e-6,
    magnitude_weight: float = 0.0,
) -> float:
    """Direction distance with an optional layer-balanced magnitude profile."""

    if magnitude_weight < 0:
        raise ValueError("Knowledge-fingerprint magnitude weight cannot be negative.")
    if set(lhs.updates) != set(rhs.updates):
        raise ValueError("Knowledge-fingerprint keys do not match.")
    direction_similarities = []
    magnitude_distances = []
    for name in sorted(lhs.updates):
        lhs_update = lhs.updates[name].float().reshape(-1)
        rhs_update = rhs.updates[name].float().reshape(-1)
        lhs_norm = torch.linalg.vector_norm(lhs_update)
        rhs_norm = torch.linalg.vector_norm(rhs_update)
        if lhs_norm <= eps and rhs_norm <= eps:
            similarity = 1.0
            magnitude_distance = 0.0
        elif lhs_norm <= eps or rhs_norm <= eps:
            similarity = 0.0
            magnitude_distance = 10.0
        else:
            similarity = float(
                torch.dot(lhs_update, rhs_update).div(lhs_norm * rhs_norm).item()
            )
            magnitude_distance = min(
                10.0,
                abs(float(torch.log((lhs_norm + eps) / (rhs_norm + eps)).item())),
            )
        direction_similarities.append(similarity)
        magnitude_distances.append(magnitude_distance)
    if not direction_similarities:
        return float("inf")
    direction_distance = 1.0 - float(
        sum(direction_similarities) / len(direction_similarities)
    )
    magnitude_distance = float(
        sum(magnitude_distances) / len(magnitude_distances)
    )
    return direction_distance + magnitude_weight * magnitude_distance


def squared_wasserstein_distance(
    lhs: DomainSignature,
    rhs: DomainSignature,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Squared W2 distance between two diagonal Gaussian signatures."""

    mean_term = torch.sum((lhs.mean - rhs.mean).square())
    lhs_std = torch.sqrt(lhs.variance.clamp_min(0) + eps)
    rhs_std = torch.sqrt(rhs.variance.clamp_min(0) + eps)
    variance_term = torch.sum((lhs_std - rhs_std).square())
    return mean_term + variance_term


def standardized_signature_distance(
    lhs: DomainSignature,
    rhs: DomainSignature,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Content-robust distance normalized by the pooled channel variance.

    Shallow single-image means are strongly affected by semantic content.  The
    pooled variance downweights channels whose activations naturally fluctuate,
    while the mean over channels makes thresholds transferable across stems.
    """

    pooled_variance = lhs.variance + rhs.variance + eps
    mean_term = ((lhs.mean - rhs.mean).square() / pooled_variance).mean()
    lhs_std = torch.sqrt(lhs.variance.clamp_min(0) + eps)
    rhs_std = torch.sqrt(rhs.variance.clamp_min(0) + eps)
    scale_term = ((lhs_std - rhs_std).square() / pooled_variance).mean()
    return mean_term + scale_term


def zoa_symmetric_kl_distance(
    lhs: DomainSignature,
    rhs: DomainSignature,
    eps: float = 1e-6,
) -> torch.Tensor:
    """ZOA's symmetric diagonal-Gaussian stem-statistic divergence."""

    lhs_variance = lhs.variance.clamp_min(eps)
    rhs_variance = rhs.variance.clamp_min(eps)
    mean_delta_squared = (lhs.mean - rhs.mean).square()
    lhs_to_rhs = (
        (lhs_variance + mean_delta_squared) / (2.0 * rhs_variance) - 0.5
    )
    rhs_to_lhs = (
        (rhs_variance + mean_delta_squared) / (2.0 * lhs_variance) - 0.5
    )
    return (lhs_to_rhs + rhs_to_lhs).mean()


def signature_distance(
    lhs: DomainSignature,
    rhs: DomainSignature,
    eps: float = 1e-6,
    mode: str = "wasserstein",
) -> torch.Tensor:
    if mode == "wasserstein":
        return squared_wasserstein_distance(lhs, rhs, eps)
    if mode == "standardized":
        return standardized_signature_distance(lhs, rhs, eps)
    if mode == "zoa_symmetric_kl":
        return zoa_symmetric_kl_distance(lhs, rhs, eps)
    raise ValueError(f"Unsupported AdaInit signature distance: {mode}")


def aggregate_signatures(
    signatures: Iterable[DomainSignature],
) -> DomainSignature:
    """Pool per-image diagonal Gaussians as if their tokens were concatenated."""

    items = list(signatures)
    if not items:
        raise ValueError("Cannot aggregate an empty signature sequence.")
    means = torch.stack([item.mean for item in items])
    variances = torch.stack([item.variance for item in items])
    pooled_mean = means.mean(dim=0)
    # E[var(X|image)] + var(E[X|image]). This is more accurate than averaging
    # only per-image variances and captures between-image variation.
    pooled_variance = (variances + means.square()).mean(dim=0) - pooled_mean.square()
    return DomainSignature(pooled_mean, pooled_variance.clamp_min(0.0))


class RollingSignatureWindow:
    """Fixed-size recent-sample signature used for detection and retrieval."""

    def __init__(self, window_size: int):
        if window_size < 1:
            raise ValueError("AdaInit signature window size must be positive.")
        self.window_size = window_size
        self._items: Deque[DomainSignature] = deque(maxlen=window_size)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def ready(self) -> bool:
        return len(self._items) == self.window_size

    def append(self, signature: DomainSignature) -> DomainSignature:
        self._items.append(signature.clone_cpu())
        return aggregate_signatures(self._items).clone_cpu()

    def reset(self) -> None:
        self._items.clear()

    def state_dict(self) -> Dict[str, Any]:
        return {"items": [item.state_dict() for item in self._items]}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        items = [
            DomainSignature.from_state_dict(item)
            for item in state.get("items", [])
        ]
        if len(items) > self.window_size:
            raise RuntimeError("Signature checkpoint exceeds configured window size.")
        self._items = deque(items, maxlen=self.window_size)


@dataclass
class DriftDecision:
    detected: bool
    score: Optional[float]
    threshold: Optional[float]
    reason: Optional[str] = None
    # ``proposal`` is deliberately separated from ``detected``.  The former
    # starts recovery evaluation at the first significant MDS deviation; the
    # latter records that the provisional MDS survived all confirmations and
    # replaced the previous main MDS.  This lets AdaInit react early without
    # pretending that every useful rollback proposal is a true domain change.
    proposal: bool = False
    transition_confirmed: bool = False
    transition_rejected: bool = False
    provisional_signature: Optional[DomainSignature] = None


class MomentumDriftDetector:
    """Momentum domain signature and segment-local EW drift statistics.

    ``signature_momentum`` is the paper's alpha: larger values retain more of
    the previous MDS. ``score_momentum`` separately controls the exponentially
    weighted mean and variance of accepted (non-shift) drift scores.
    """

    def __init__(
        self,
        signature_momentum: float,
        score_momentum: float,
        threshold_beta: float,
        eps: float,
        min_reference_samples: int,
        required_confirmations: int = 1,
        mode: str = "adaptive",
        periodic_interval: int = 0,
        cooldown_samples: int = 0,
        distance_mode: str = "wasserstein",
        minimum_exceedance_ratio: float = 1.0,
        fixed_threshold: Optional[float] = None,
    ):
        if not 0 <= signature_momentum < 1:
            raise ValueError("signature_momentum must be in [0, 1).")
        if not 0 <= score_momentum < 1:
            raise ValueError("score_momentum must be in [0, 1).")
        if min_reference_samples < 1:
            raise ValueError("min_reference_samples must be positive.")
        if required_confirmations < 1:
            raise ValueError("required_confirmations must be positive.")
        if mode not in {"adaptive", "fixed", "disabled", "periodic", "hybrid"}:
            raise ValueError(f"Unsupported AdaInit detector mode: {mode}")
        if mode in {"periodic", "hybrid"} and periodic_interval < 1:
            raise ValueError(
                f"{mode} detector mode requires a positive interval."
            )
        if cooldown_samples < 0:
            raise ValueError("AdaInit trigger cooldown cannot be negative.")
        if distance_mode not in {
            "wasserstein",
            "standardized",
            "zoa_symmetric_kl",
        }:
            raise ValueError(f"Unsupported AdaInit signature distance: {distance_mode}")
        if minimum_exceedance_ratio < 1.0:
            raise ValueError("Drift minimum exceedance ratio must be at least one.")
        if mode == "fixed" and (fixed_threshold is None or fixed_threshold <= 0):
            raise ValueError("Fixed drift detection requires a positive threshold.")

        self.signature_momentum = signature_momentum
        self.score_momentum = score_momentum
        self.threshold_beta = threshold_beta
        self.eps = eps
        self.min_reference_samples = min_reference_samples
        self.required_confirmations = required_confirmations
        self.mode = mode
        self.periodic_interval = periodic_interval
        self.cooldown_samples = cooldown_samples
        self.distance_mode = distance_mode
        self.minimum_exceedance_ratio = minimum_exceedance_ratio
        self.fixed_threshold = fixed_threshold
        self.reset()

    def reset(self) -> None:
        self.signature: Optional[DomainSignature] = None
        self.score_mean: Optional[float] = None
        self.score_variance: float = 0.0
        self.score_count: int = 0
        self.segment_length: int = 0
        self.pending_signatures: List[DomainSignature] = []
        self.pending_scores: List[float] = []
        self.last_detection_index: Optional[int] = None

    def _update_score_statistics(self, score: float) -> None:
        if self.score_mean is None:
            self.score_mean = score
            self.score_variance = 0.0
        else:
            # EW variance update using the pre-update and post-update means.
            previous_mean = self.score_mean
            updated_mean = (
                self.score_momentum * previous_mean
                + (1.0 - self.score_momentum) * score
            )
            self.score_variance = (
                self.score_momentum * self.score_variance
                + (1.0 - self.score_momentum)
                * (score - previous_mean)
                * (score - updated_mean)
            )
            self.score_mean = updated_mean
        self.score_count += 1

    def _update_signature(self, current: DomainSignature) -> None:
        assert self.signature is not None
        alpha = self.signature_momentum
        self.signature = DomainSignature(
            mean=alpha * self.signature.mean + (1.0 - alpha) * current.mean,
            variance=(
                alpha * self.signature.variance
                + (1.0 - alpha) * current.variance
            ),
        )

    def observe(
        self, current: DomainSignature, stream_index: int
    ) -> DriftDecision:
        """Update the two-stage main/provisional MDS state machine.

        A first threshold exceedance freezes the main MDS and emits a recovery
        proposal immediately.  Subsequent exceedances update only a provisional
        MDS.  Once the configured confirmation count is reached, that
        provisional MDS becomes the main MDS.  If the run breaks, all withheld
        observations are replayed into the original MDS, so an isolated content
        outlier does not silently remove samples from the reference trajectory.
        """

        current = current.clone_cpu()
        if self.signature is None:
            self.signature = current
            self.segment_length = 1
            return DriftDecision(False, None, None)

        score = float(
            signature_distance(
                current,
                self.signature,
                self.eps,
                self.distance_mode,
            ).item()
        )
        std = self.score_variance ** 0.5
        threshold = (
            self.fixed_threshold
            if self.mode == "fixed"
            else (
                None
                if self.score_mean is None
                else self.score_mean + self.threshold_beta * std
            )
        )

        in_cooldown = bool(
            self.last_detection_index is not None
            and stream_index - self.last_detection_index <= self.cooldown_samples
        )
        periodic_detected = bool(
            self.mode in {"periodic", "hybrid"}
            and stream_index > 0
            and stream_index % self.periodic_interval == 0
            and not in_cooldown
        )
        reason = None
        if self.mode == "disabled":
            self._update_score_statistics(score)
            self._update_signature(current)
            self.segment_length += 1
            return DriftDecision(False, score, threshold)
        elif self.mode == "periodic":
            # A periodic event is a recovery proposal, not a claim that the
            # active domain changed.  It therefore never replaces the main MDS.
            self._update_score_statistics(score)
            self._update_signature(current)
            self.segment_length += 1
            return DriftDecision(
                periodic_detected,
                score,
                threshold,
                "periodic" if periodic_detected else None,
                proposal=periodic_detected,
            )
        else:
            reference_ready = (
                self.mode == "fixed"
                or self.score_count >= self.min_reference_samples
            )
            threshold_exceeded = bool(
                reference_ready
                and threshold is not None
                and score > threshold * self.minimum_exceedance_ratio
                and not in_cooldown
            )
            if threshold_exceeded:
                proposal_started = not self.pending_signatures
                self.pending_signatures.append(current)
                self.pending_scores.append(score)
                provisional = aggregate_signatures(
                    self.pending_signatures
                ).clone_cpu()
                adaptive_confirmed = (
                    len(self.pending_signatures) >= self.required_confirmations
                )
                if adaptive_confirmed:
                    # Promote only the provisional MDS.  Triggering scores were
                    # measured against the old MDS and cannot calibrate the new
                    # segment's threshold, so its score statistics restart.
                    self.signature = provisional
                    self.score_mean = None
                    self.score_variance = 0.0
                    self.score_count = 0
                    self.segment_length = len(self.pending_signatures)
                    self.pending_signatures = []
                    self.pending_scores = []
                    self.last_detection_index = stream_index
                    return DriftDecision(
                        True,
                        score,
                        threshold,
                        "adaptive",
                        proposal=proposal_started,
                        transition_confirmed=True,
                        provisional_signature=provisional,
                    )
                return DriftDecision(
                    periodic_detected,
                    score,
                    threshold,
                    "adaptive" if proposal_started else (
                        "periodic" if periodic_detected else None
                    ),
                    proposal=proposal_started or periodic_detected,
                    provisional_signature=provisional,
                )

            transition_rejected = bool(self.pending_signatures)
            if transition_rejected:
                # Restore continuity of the main MDS after a rejected proposal.
                # The withheld run is replayed in causal order, followed by the
                # current observation that broke the exceedance sequence.
                for pending_signature, pending_score in zip(
                    self.pending_signatures, self.pending_scores
                ):
                    self._update_score_statistics(pending_score)
                    self._update_signature(pending_signature)
                    self.segment_length += 1
                self.pending_signatures = []
                self.pending_scores = []

            self._update_score_statistics(score)
            self._update_signature(current)
            self.segment_length += 1
            return DriftDecision(
                periodic_detected,
                score,
                threshold,
                "periodic" if periodic_detected else None,
                proposal=periodic_detected,
                transition_rejected=transition_rejected,
            )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "signature": (
                None if self.signature is None else self.signature.state_dict()
            ),
            "score_mean": self.score_mean,
            "score_variance": self.score_variance,
            "score_count": self.score_count,
            "segment_length": self.segment_length,
            "pending_signatures": [
                item.state_dict() for item in self.pending_signatures
            ],
            "pending_scores": self.pending_scores,
            "last_detection_index": self.last_detection_index,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        signature = state.get("signature")
        self.signature = (
            None if signature is None else DomainSignature.from_state_dict(signature)
        )
        self.score_mean = state.get("score_mean")
        self.score_variance = state.get("score_variance", 0.0)
        self.score_count = state.get("score_count", 0)
        self.segment_length = state.get("segment_length", 0)
        self.pending_signatures = [
            DomainSignature.from_state_dict(item)
            for item in state.get("pending_signatures", [])
        ]
        self.pending_scores = list(state.get("pending_scores", []))
        self.last_detection_index = state.get("last_detection_index")


@dataclass
class InitializationCacheEntry:
    adaptable_state: Dict[str, torch.Tensor]
    signature: Optional[DomainSignature]
    insertion_time: int
    optimizer_state: Optional[Dict[str, Any]] = None
    backend_runtime_state: Optional[Dict[str, Any]] = None
    match_radius: Optional[float] = None
    knowledge_fingerprint: Optional[KnowledgeFingerprint] = None
    health_score: Optional[float] = None
    health_details: Optional[Dict[str, float]] = None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "adaptable_state": self.adaptable_state,
            "signature": (
                None if self.signature is None else self.signature.state_dict()
            ),
            "insertion_time": self.insertion_time,
            "optimizer_state": self.optimizer_state,
            "backend_runtime_state": self.backend_runtime_state,
            "match_radius": self.match_radius,
            "knowledge_fingerprint": (
                None
                if self.knowledge_fingerprint is None
                else self.knowledge_fingerprint.state_dict()
            ),
            "health_score": self.health_score,
            "health_details": self.health_details,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "InitializationCacheEntry":
        signature = state.get("signature")
        knowledge_fingerprint = state.get("knowledge_fingerprint")
        return cls(
            adaptable_state=state["adaptable_state"],
            signature=(
                None
                if signature is None
                else DomainSignature.from_state_dict(signature)
            ),
            insertion_time=state["insertion_time"],
            optimizer_state=state.get("optimizer_state"),
            backend_runtime_state=state.get("backend_runtime_state"),
            match_radius=state.get("match_radius"),
            knowledge_fingerprint=(
                None
                if knowledge_fingerprint is None
                else KnowledgeFingerprint.from_state_dict(
                    knowledge_fingerprint
                )
            ),
            health_score=state.get("health_score"),
            health_details=state.get("health_details"),
        )


class InitializationCache:
    """Bounded adaptable-state cache with closest-pair/older eviction."""

    def __init__(
        self,
        capacity: int,
        eps: float = 1e-6,
        distance_mode: str = "wasserstein",
        eviction_mode: str = "signature",
        reference_state: Optional[Dict[str, torch.Tensor]] = None,
        fingerprint_magnitude_weight: float = 0.0,
    ):
        if capacity < 0:
            raise ValueError("AdaInit cache capacity cannot be negative.")
        self.capacity = capacity
        self.eps = eps
        self.distance_mode = distance_mode
        if eviction_mode not in {
            "signature",
            "parameter_cosine",
            "knowledge_fingerprint",
        }:
            raise ValueError(
                f"Unsupported AdaInit cache eviction mode: {eviction_mode}"
            )
        if eviction_mode == "parameter_cosine" and reference_state is None:
            raise ValueError(
                "Parameter-cosine cache eviction requires a source reference state."
            )
        self.eviction_mode = eviction_mode
        self.reference_state = clone_state_to_cpu(reference_state)
        if fingerprint_magnitude_weight < 0:
            raise ValueError(
                "Knowledge-fingerprint magnitude weight cannot be negative."
            )
        self.fingerprint_magnitude_weight = fingerprint_magnitude_weight
        self.entries: List[InitializationCacheEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _clone_adaptable_state(
        adaptable_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in adaptable_state.items()
        }

    def nearest(
        self, query: DomainSignature
    ) -> Optional[InitializationCacheEntry]:
        entry, _, _ = self.nearest_with_distance(query)
        return entry

    def nearest_with_distance(
        self, query: DomainSignature
    ) -> Tuple[Optional[InitializationCacheEntry], Optional[float], Optional[float]]:
        entry, distance, ratio, _ = self.nearest_with_match(query)
        return entry, distance, ratio

    def nearest_with_match(
        self, query: DomainSignature
    ) -> Tuple[
        Optional[InitializationCacheEntry],
        Optional[float],
        Optional[float],
        Optional[float],
    ]:
        if not self.entries:
            return None, None, None, None
        entries = [entry for entry in self.entries if entry.signature is not None]
        if not entries:
            return None, None, None, None
        distances = [
            float(
                signature_distance(
                    entry.signature,
                    query,
                    self.eps,
                    self.distance_mode,
                ).item()
            )
            for entry in entries
        ]
        nearest_index = min(range(len(distances)), key=distances.__getitem__)
        median_distance = float(torch.tensor(distances).median().item())
        ratio = distances[nearest_index] / max(median_distance, self.eps)
        entry = entries[nearest_index]
        calibrated_ratio = (
            None
            if entry.match_radius is None
            else distances[nearest_index] / max(entry.match_radius, self.eps)
        )
        return entry, distances[nearest_index], ratio, calibrated_ratio

    def ranked_with_match(
        self, query: DomainSignature
    ) -> List[Tuple[InitializationCacheEntry, float, float, Optional[float]]]:
        """Return every cache entry ordered by signature proximity."""

        entries = [entry for entry in self.entries if entry.signature is not None]
        if not entries:
            return []
        distances = [
            float(
                signature_distance(
                    entry.signature,
                    query,
                    self.eps,
                    self.distance_mode,
                ).item()
            )
            for entry in entries
        ]
        median_distance = float(torch.tensor(distances).median().item())
        ranked = []
        for entry, distance in zip(entries, distances):
            ratio = distance / max(median_distance, self.eps)
            calibrated_ratio = (
                None
                if entry.match_radius is None
                else distance / max(entry.match_radius, self.eps)
            )
            ranked.append((entry, distance, ratio, calibrated_ratio))
        return sorted(ranked, key=lambda item: item[1])

    def ranked_by_knowledge_fingerprint(
        self,
        query: KnowledgeFingerprint,
    ) -> List[Tuple[InitializationCacheEntry, float, float]]:
        """Return cache entries ordered by local adaptation-response distance."""

        entries = [
            entry
            for entry in self.entries
            if entry.knowledge_fingerprint is not None
        ]
        if not entries:
            return []
        distances = [
            knowledge_fingerprint_distance(
                entry.knowledge_fingerprint,
                query,
                eps=self.eps,
                magnitude_weight=self.fingerprint_magnitude_weight,
            )
            for entry in entries
        ]
        median_distance = float(torch.tensor(distances).median().item())
        ranked = [
            (entry, distance, distance / max(median_distance, self.eps))
            for entry, distance in zip(entries, distances)
        ]
        return sorted(ranked, key=lambda item: item[1])

    def insert(
        self,
        adaptable_state: Dict[str, torch.Tensor],
        signature: Optional[DomainSignature],
        insertion_time: int,
        optimizer_state: Optional[Dict[str, Any]] = None,
        backend_runtime_state: Optional[Dict[str, Any]] = None,
        match_radius: Optional[float] = None,
        knowledge_fingerprint: Optional[KnowledgeFingerprint] = None,
        health_score: Optional[float] = None,
        health_details: Optional[Dict[str, float]] = None,
    ) -> None:
        if self.capacity == 0:
            return
        self.entries.append(
            InitializationCacheEntry(
                adaptable_state=self._clone_adaptable_state(adaptable_state),
                signature=(
                    None if signature is None else signature.clone_cpu()
                ),
                insertion_time=insertion_time,
                optimizer_state=clone_state_to_cpu(optimizer_state),
                backend_runtime_state=clone_state_to_cpu(backend_runtime_state),
                match_radius=(
                    None if match_radius is None else float(match_radius)
                ),
                knowledge_fingerprint=(
                    None
                    if knowledge_fingerprint is None
                    else knowledge_fingerprint.clone_cpu()
                ),
                health_score=(
                    None if health_score is None else float(health_score)
                ),
                health_details=(
                    None
                    if health_details is None
                    else {
                        name: float(value)
                        for name, value in health_details.items()
                    }
                ),
            )
        )
        if len(self.entries) > self.capacity:
            self._evict_redundant_entry()

    def _evict_redundant_entry(self) -> None:
        closest_pair = None
        best_pair_score = (
            -float("inf")
            if self.eviction_mode == "parameter_cosine"
            else float("inf")
        )
        for left in range(len(self.entries)):
            for right in range(left + 1, len(self.entries)):
                if self.eviction_mode == "signature":
                    if (
                        self.entries[left].signature is None
                        or self.entries[right].signature is None
                    ):
                        continue
                    pair_score = float(
                        signature_distance(
                            self.entries[left].signature,
                            self.entries[right].signature,
                            self.eps,
                            self.distance_mode,
                        ).item()
                    )
                    is_better = pair_score < best_pair_score
                elif self.eviction_mode == "parameter_cosine":
                    pair_score = self._adaptation_vector_similarity(
                        self.entries[left].adaptable_state,
                        self.entries[right].adaptable_state,
                    )
                    is_better = pair_score > best_pair_score
                else:
                    lhs_fingerprint = self.entries[left].knowledge_fingerprint
                    rhs_fingerprint = self.entries[right].knowledge_fingerprint
                    if lhs_fingerprint is None or rhs_fingerprint is None:
                        continue
                    pair_score = knowledge_fingerprint_distance(
                        lhs_fingerprint,
                        rhs_fingerprint,
                        eps=self.eps,
                        magnitude_weight=self.fingerprint_magnitude_weight,
                    )
                    is_better = pair_score < best_pair_score
                if is_better:
                    best_pair_score = pair_score
                    closest_pair = (left, right)

        if closest_pair is None:
            # Backward-compatible checkpoints may lack the metadata required by
            # a newly selected eviction rule.  FIFO is deterministic and safe.
            oldest_index = min(
                range(len(self.entries)),
                key=lambda index: self.entries[index].insertion_time,
            )
            del self.entries[oldest_index]
            return
        left, right = closest_pair
        left_health = self.entries[left].health_score
        right_health = self.entries[right].health_score
        if (
            self.eviction_mode == "knowledge_fingerprint"
            and left_health is not None
            and right_health is not None
            and abs(left_health - right_health) > self.eps
        ):
            remove_index = left if left_health < right_health else right
        else:
            remove_index = (
                left
                if self.entries[left].insertion_time
                <= self.entries[right].insertion_time
                else right
            )
        del self.entries[remove_index]

    def _adaptation_vector_similarity(
        self,
        lhs: Dict[str, torch.Tensor],
        rhs: Dict[str, torch.Tensor],
    ) -> float:
        """ZOA-style mean layer cosine between source-relative updates."""

        similarities = []
        for name in sorted(self.reference_state):
            reference = self.reference_state[name].float().reshape(-1)
            lhs_delta = lhs[name].float().reshape(-1) - reference
            rhs_delta = rhs[name].float().reshape(-1) - reference
            lhs_norm = torch.linalg.vector_norm(lhs_delta)
            rhs_norm = torch.linalg.vector_norm(rhs_delta)
            if lhs_norm <= self.eps and rhs_norm <= self.eps:
                similarity = 1.0
            elif lhs_norm <= self.eps or rhs_norm <= self.eps:
                similarity = 0.0
            else:
                similarity = float(
                    torch.dot(lhs_delta, rhs_delta).div(lhs_norm * rhs_norm).item()
                )
            similarities.append(similarity)
        return float(sum(similarities) / len(similarities))

    def state_dict(self) -> Dict[str, Any]:
        return {"entries": [entry.state_dict() for entry in self.entries]}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.entries = [
            InitializationCacheEntry.from_state_dict(entry_state)
            for entry_state in state.get("entries", [])
        ]
        if len(self.entries) > self.capacity:
            raise RuntimeError(
                "Recovery checkpoint contains more AdaInit cache entries than "
                "the configured capacity."
            )
