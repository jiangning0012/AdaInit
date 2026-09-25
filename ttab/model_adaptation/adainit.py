# -*- coding: utf-8 -*-
"""AdaInit: recovery-aware initialization for single-sample CTTA."""

import copy
import functools
import random
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ttab.api import Batch
from ttab.loads.datasets.imagenet.data_aug_imagenet import aug_imagenet
from ttab.model_adaptation.adainit_backends import build_adainit_backend
from ttab.model_adaptation.adainit_components import (
    build_knowledge_fingerprint,
    clone_state_to_cpu,
    DomainSignature,
    DriftDecision,
    InitializationCache,
    KnowledgeFingerprint,
    knowledge_fingerprint_similarity,
    knowledge_fingerprint_statistics,
    MomentumDriftDetector,
    RollingSignatureWindow,
    signature_distance,
)
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.auxiliary import fork_rng_with_seed
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer


@dataclass
class InitializationCandidate:
    name: str
    adaptable_state: Dict[str, torch.Tensor]
    signature_distance: Optional[float] = None
    signature_distance_ratio: Optional[float] = None
    optimizer_state: Optional[Dict[str, Any]] = None
    backend_runtime_state: Optional[Dict[str, Any]] = None
    cache_insertion_time: Optional[int] = None
    signature_calibrated_ratio: Optional[float] = None
    knowledge_distance: Optional[float] = None
    knowledge_distance_ratio: Optional[float] = None
    cache_health_score: Optional[float] = None


@dataclass
class CausalEvidenceBranch:
    """One independently evolved hard candidate used only for selection evidence."""

    candidate: InitializationCandidate
    optimizer_state: Dict[str, Any]
    backend_runtime_state: Dict[str, Any]
    entropy_sum: float = 0.0
    view_jsd_sum: float = 0.0
    observations: int = 0


@dataclass
class PendingHardSelection:
    """Causal multi-frame evidence for one non-overlapping hard decision."""

    trigger_index: int
    trigger_reason: str
    drift_strength_ratio: Optional[float]
    branches: List[CausalEvidenceBranch]
    processed_samples: int = 0


@dataclass
class SequentialEvidenceBranch:
    """One hard initialization trajectory in the sequential safety test."""

    candidate: InitializationCandidate
    optimizer_state: Dict[str, Any]
    backend_runtime_state: Dict[str, Any]
    marginal_entropies: Deque[float]
    view_jsds: Deque[float]
    marginal_probabilities: Deque[torch.Tensor]
    observations: int = 0


@dataclass
class PendingSequentialSelection:
    """Causal recovery proposal with a provisional MDS and delayed branches."""

    trigger_index: int
    trigger_reason: str
    drift_strength_ratio: Optional[float]
    trigger_current: InitializationCandidate
    proposal_signature: Optional[DomainSignature]
    include_source: bool
    buffered_inputs: List[torch.Tensor]
    branches: List[SequentialEvidenceBranch]
    processed_samples: int = 0
    transition_status: str = "pending"


class AdaInit(BaseAdaptation):
    """Select an adaptable initialization when a drift is detected.

    AdaInit is an adaptation method (an initialization controller around a TTA
    backend), not a TTAB model-selection method.  TTAB's model-selection layer
    remains responsible only for selecting among update-step checkpoints.
    """

    def __init__(self, meta_conf, model: nn.Module):
        # BaseAdaptation invokes the initialization hooks during super().__init__,
        # so the backend must exist before that call.
        self._backend = build_adainit_backend(meta_conf)
        super().__init__(meta_conf, model)
        self._backend.meta_conf = self._meta_conf

        self._source_adaptable_state = self._capture_adaptable_state()
        self._backend_runtime_state = self._backend.fresh_runtime_state()
        self._initialize_native_anchor()
        self._detector = self._build_detector()
        self._initialization_cache = InitializationCache(
            capacity=self._meta_conf.adainit_cache_size,
            eps=self._meta_conf.adainit_eps,
            distance_mode=self._meta_conf.adainit_signature_distance,
            eviction_mode=self._meta_conf.adainit_cache_eviction,
            reference_state=self._source_adaptable_state,
            fingerprint_magnitude_weight=(
                self._meta_conf.adainit_fingerprint_magnitude_weight
            ),
        )
        self._signature_window = RollingSignatureWindow(
            self._meta_conf.adainit_signature_window_size
        )
        self._selection_buffer = deque(
            maxlen=self._meta_conf.adainit_selection_window
        )
        # This label queue is populated only by an explicit diagnostic flag and
        # is read only after candidate scores have been computed. It must never
        # influence drift detection, signatures, counterfactual adaptation, or
        # initialization selection.
        self._selection_audit_labels = deque(
            maxlen=self._meta_conf.adainit_selection_window
        )
        self._knowledge_prediction_window: Deque[torch.Tensor] = deque(
            maxlen=self._meta_conf.adainit_fingerprint_window
        )
        self._knowledge_state_snapshots = deque(maxlen=4)
        self._knowledge_state_snapshots.append(
            (0, clone_state_to_cpu(self._source_adaptable_state))
        )
        # Captured once when a run of adaptive evidence starts.  If the run is
        # confirmed, this is theta immediately before the suspected shift—not
        # the already-contaminated state several confirmation samples later.
        self._pending_pre_shift_state: Optional[InitializationCandidate] = None
        self._pending_hard_selection: Optional[PendingHardSelection] = None
        self._pending_sequential_selection: Optional[
            PendingSequentialSelection
        ] = None
        self._transform_helper = self._get_transform_helper()
        self._stream_index = 0
        self._number_of_detections = 0
        self._candidate_selection_counts = {
            "current": 0,
            "source": 0,
            "history": 0,
            "native_anchor": 0,
        }
        self._oracle_lookahead_batches: List[Batch] = []
        self._last_oracle_score_details: Dict[str, Dict[str, float]] = {}
        self._last_oracle_surrogate_details: Dict[str, Dict[str, float]] = {}
        self._last_causal_evidence_details: Dict[str, Dict[str, float]] = {}
        self._last_sequential_evidence_details: Dict[str, Dict[str, float]] = {}

    def _prior_safety_check(self):
        super()._prior_safety_check()
        if self._meta_conf.data_wise != "sample_wise":
            raise ValueError("AdaInit currently implements the single-sample protocol only.")
        if self._meta_conf.batch_size != 1:
            raise ValueError("AdaInit requires batch_size=1.")
        if self._meta_conf.episodic:
            raise ValueError(
                "AdaInit requires a continual stream; use --episodic false."
            )
        if self._meta_conf.stochastic_restore_model:
            raise ValueError(
                "Stochastic restoration changes AdaInit's cached formal state and "
                "must be disabled for the paper protocol."
            )
        if self._meta_conf.fishers:
            raise ValueError(
                "AdaInit backends do not apply the optional TTAB Fisher penalty; "
                "use --fishers false for a faithful comparison."
            )
        if self._meta_conf.adainit_num_views < 1:
            raise ValueError("AdaInit requires at least one transformed view.")
        if (
            self._meta_conf.adainit_recovery_score == "view_jsd"
            and self._meta_conf.adainit_num_views < 2
        ):
            raise ValueError("AdaInit view_jsd scoring requires at least two views.")
        if self._meta_conf.adainit_counterfactual_steps < 1:
            raise ValueError("AdaInit requires at least one counterfactual step.")
        if self._meta_conf.adainit_selection_margin < 0:
            raise ValueError("AdaInit selection margin cannot be negative.")
        if self._meta_conf.adainit_source_selection_margin < 0:
            raise ValueError("AdaInit source selection margin cannot be negative.")
        if self._meta_conf.adainit_native_anchor_selection_margin < 0:
            raise ValueError(
                "AdaInit native-anchor selection margin cannot be negative."
            )
        if self._meta_conf.adainit_cache_insert_interval < 1:
            raise ValueError("AdaInit cache insertion interval must be positive.")
        if self._meta_conf.adainit_cache_min_segment_samples < 0:
            raise ValueError("AdaInit cache minimum segment age cannot be negative.")
        if self._meta_conf.adainit_drift_confirmations < 1:
            raise ValueError("AdaInit drift confirmations must be positive.")
        if self._meta_conf.adainit_signature_window_size < 1:
            raise ValueError("AdaInit signature window size must be positive.")
        if self._meta_conf.adainit_selection_window < 1:
            raise ValueError("AdaInit selection window must be positive.")
        if self._meta_conf.adainit_candidate_eval_batch_size < 0:
            raise ValueError("AdaInit candidate evaluation batch size cannot be negative.")
        if self._meta_conf.adainit_oracle_horizon < 0:
            raise ValueError("AdaInit oracle horizon cannot be negative.")
        if self._meta_conf.adainit_evidence_horizon < 1:
            raise ValueError("AdaInit causal evidence horizon must be positive.")
        if self._meta_conf.adainit_evidence_min_samples < 1:
            raise ValueError("AdaInit evidence minimum must be positive.")
        if (
            self._meta_conf.adainit_evidence_min_samples
            > self._meta_conf.adainit_evidence_horizon
        ):
            raise ValueError(
                "AdaInit evidence minimum cannot exceed its rolling window."
            )
        if (
            self._meta_conf.adainit_evidence_max_samples
            < self._meta_conf.adainit_evidence_min_samples
        ):
            raise ValueError(
                "AdaInit evidence maximum cannot be smaller than its minimum."
            )
        if self._meta_conf.adainit_evidence_confidence_scale < 0:
            raise ValueError(
                "AdaInit evidence confidence scale cannot be negative."
            )
        if not (
            0
            <= self._meta_conf.adainit_evidence_warmup
            < self._meta_conf.adainit_evidence_horizon
        ):
            raise ValueError(
                "AdaInit evidence warmup must be in [0, evidence_horizon)."
            )
        if self._meta_conf.adainit_oracle_max_history_candidates < 0:
            raise ValueError(
                "AdaInit oracle maximum history candidates cannot be negative."
            )
        if self._meta_conf.adainit_max_history_candidates < 0:
            raise ValueError(
                "AdaInit maximum history candidates cannot be negative."
            )
        if self._meta_conf.adainit_history_min_age < 0:
            raise ValueError("AdaInit history minimum age cannot be negative.")
        if self._meta_conf.adainit_oracle_min_accuracy_gain < 0:
            raise ValueError("AdaInit oracle accuracy margin cannot be negative.")
        if (
            self._meta_conf.adainit_initialization_selector
            == "oracle_future_accuracy"
            and self._meta_conf.adainit_oracle_horizon < 1
        ):
            raise ValueError(
                "The future-accuracy oracle requires --adainit_oracle_horizon >= 1."
            )
        if self._meta_conf.adainit_trigger_cooldown < 0:
            raise ValueError("AdaInit trigger cooldown cannot be negative.")
        if self._meta_conf.adainit_drift_min_ratio < 1:
            raise ValueError("AdaInit drift minimum ratio must be at least one.")
        if self._meta_conf.adainit_history_score_bonus < 0:
            raise ValueError("AdaInit history score bonus cannot be negative.")
        if self._meta_conf.adainit_history_min_drift_ratio < 0:
            raise ValueError(
                "AdaInit history minimum drift ratio cannot be negative."
            )
        if self._meta_conf.adainit_context_diversity_weight < 0:
            raise ValueError("AdaInit context diversity weight cannot be negative.")
        if self._meta_conf.adainit_source_min_drift_ratio < 1:
            raise ValueError("AdaInit source drift ratio must be at least one.")
        if self._meta_conf.adainit_periodic_selection_margin < 0:
            raise ValueError("AdaInit periodic selection margin cannot be negative.")
        if self._meta_conf.adainit_fingerprint_window < 2:
            raise ValueError("AdaInit fingerprint window must be at least two.")
        if self._meta_conf.adainit_fingerprint_window % 2 != 0:
            raise ValueError("AdaInit fingerprint window must be even.")
        if self._meta_conf.adainit_fingerprint_magnitude_weight < 0:
            raise ValueError(
                "AdaInit fingerprint magnitude weight cannot be negative."
            )
        if not (
            0.0
            <= self._meta_conf.adainit_cache_health_max_concentration
            <= 1.0
        ):
            raise ValueError(
                "AdaInit cache health concentration must be in [0, 1]."
            )
        if self._meta_conf.adainit_cache_health_min_information < 0:
            raise ValueError(
                "AdaInit cache health minimum information cannot be negative."
            )
        if not (
            -1.0
            <= self._meta_conf.adainit_cache_health_min_update_cosine
            <= 1.0
        ):
            raise ValueError(
                "AdaInit cache health update cosine must be in [-1, 1]."
            )
        if self._meta_conf.adainit_cache_health_min_fingerprint_norm < 0:
            raise ValueError(
                "AdaInit cache health minimum fingerprint norm cannot be negative."
            )
        if self._meta_conf.adainit_cache_health_update_weight < 0:
            raise ValueError(
                "AdaInit cache health update weight cannot be negative."
            )
        if self._meta_conf.adainit_sequential_view_jsd_weight < 0:
            raise ValueError("AdaInit sequential View JSD weight cannot be negative.")
        if self._meta_conf.adainit_sequential_context_weight < 0:
            raise ValueError(
                "AdaInit sequential context weight cannot be negative."
            )
        if (
            self._meta_conf.adainit_history_retrieval
            == "knowledge_fingerprint"
            and self._meta_conf.adainit_knowledge_fingerprint == "disabled"
        ):
            raise ValueError(
                "Knowledge-fingerprint retrieval requires a knowledge fingerprint."
            )
        if (
            self._meta_conf.adainit_cache_eviction == "knowledge_fingerprint"
            and self._meta_conf.adainit_knowledge_fingerprint == "disabled"
        ):
            raise ValueError(
                "Knowledge-fingerprint eviction requires a knowledge fingerprint."
            )
        if (
            self._meta_conf.adainit_cache_admission == "periodic_health"
            and (
                self._meta_conf.adainit_knowledge_fingerprint == "disabled"
                or self._meta_conf.adainit_history_retrieval
                != "knowledge_fingerprint"
            )
        ):
            raise ValueError(
                "Periodic-health cache admission requires knowledge-fingerprint "
                "storage and retrieval."
            )

    def _initialize_model(self, model: nn.Module):
        return self._backend.configure_model(model, self._meta_conf.device)

    def _initialize_trainable_parameters(self):
        params, names, module_names = self._backend.collect_parameters(self._model)
        self._adapt_params = params
        self._adapt_param_names = names
        self._adapt_module_names = module_names
        return params, names

    def _initialize_optimizer(self, params) -> torch.optim.Optimizer:
        return self._backend.build_optimizer(self._meta_conf, params)

    def _build_detector(self) -> MomentumDriftDetector:
        detector_distance = self._meta_conf.adainit_detector_distance
        if detector_distance == "same":
            detector_distance = self._meta_conf.adainit_signature_distance
        return MomentumDriftDetector(
            signature_momentum=self._meta_conf.adainit_signature_momentum,
            score_momentum=self._meta_conf.adainit_score_momentum,
            threshold_beta=self._meta_conf.adainit_drift_beta,
            eps=self._meta_conf.adainit_eps,
            min_reference_samples=self._meta_conf.adainit_min_reference_samples,
            required_confirmations=self._meta_conf.adainit_drift_confirmations,
            mode=self._meta_conf.adainit_detector_mode,
            periodic_interval=self._meta_conf.adainit_periodic_interval,
            cooldown_samples=self._meta_conf.adainit_trigger_cooldown,
            distance_mode=detector_distance,
            minimum_exceedance_ratio=self._meta_conf.adainit_drift_min_ratio,
            fixed_threshold=self._meta_conf.adainit_fixed_drift_threshold,
        )

    def _get_transform_helper(self):
        if self._meta_conf.base_data_name == "imagenet":
            return aug_imagenet
        raise NotImplementedError(
            f"AdaInit has no view transform for {self._meta_conf.base_data_name}."
        )

    def _capture_adaptable_state(self) -> Dict[str, torch.Tensor]:
        named_parameters = dict(self._model.named_parameters())
        return {
            name: named_parameters[name].detach().cpu().clone()
            for name in self._adapt_param_names
        }

    def _load_adaptable_state(self, state: Dict[str, torch.Tensor]) -> None:
        if set(state) != set(self._adapt_param_names):
            missing = sorted(set(self._adapt_param_names) - set(state))
            unexpected = sorted(set(state) - set(self._adapt_param_names))
            raise RuntimeError(
                "AdaInit adaptable-state mismatch: "
                f"missing={missing}, unexpected={unexpected}."
            )
        named_parameters = dict(self._model.named_parameters())
        with torch.no_grad():
            for name in self._adapt_param_names:
                named_parameters[name].copy_(
                    state[name].to(
                        device=named_parameters[name].device,
                        dtype=named_parameters[name].dtype,
                    )
                )

    def _reset_backend_optimizer(
        self,
        optimizer_state: Optional[Dict[str, Any]] = None,
        runtime_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._adapt_params = [
            dict(self._model.named_parameters())[name]
            for name in self._adapt_param_names
        ]
        self._optimizer = self._backend.build_optimizer(
            self._meta_conf, self._adapt_params
        )
        if optimizer_state is not None:
            self._optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        self._backend_runtime_state = (
            self._backend.fresh_runtime_state()
            if runtime_state is None
            else copy.deepcopy(runtime_state)
        )

    def _initialize_native_anchor(self) -> None:
        """Create the optional uninterrupted backend parameter trajectory.

        The dedicated optimizer is used only to evolve the anchor itself.  An
        anchor selection loads only its model parameters and starts a fresh
        formal optimizer, preserving AdaInit's hard-initialization boundary.
        """

        if not self._meta_conf.adainit_native_anchor:
            self._native_anchor_state = None
            self._native_anchor_optimizer = None
            self._native_anchor_runtime_state = None
            return
        self._native_anchor_state = clone_state_to_cpu(
            self._source_adaptable_state
        )
        self._native_anchor_optimizer = self._backend.build_optimizer(
            self._meta_conf, self._adapt_params
        )
        self._native_anchor_runtime_state = self._backend.fresh_runtime_state()

    def _advance_native_anchor(
        self, batch: Batch, number_of_steps: int, timer: Timer
    ) -> None:
        """Advance the anchor on x_t without mutating the formal trajectory."""

        if self._native_anchor_state is None:
            return
        formal_state = self._capture_adaptable_state()
        self._load_adaptable_state(self._native_anchor_state)
        try:
            for _ in range(number_of_steps):
                result = self._backend.adapt_step(
                    self._model,
                    self._native_anchor_optimizer,
                    batch,
                    self._native_anchor_runtime_state,
                    timer,
                    random_seed=self._meta_conf.seed,
                )
                if result.reset_requested:
                    self._load_adaptable_state(self._source_adaptable_state)
                    self._native_anchor_optimizer = self._backend.build_optimizer(
                        self._meta_conf, self._adapt_params
                    )
            self._native_anchor_state = self._capture_adaptable_state()
        finally:
            self._load_adaptable_state(formal_state)

    def reset(self):
        """Restore the entire online AdaInit state (used for explicit restarts)."""

        self._model.load_state_dict(self.model_state_dict)
        self._source_adaptable_state = self._capture_adaptable_state()
        self._reset_backend_optimizer()
        self._initialize_native_anchor()
        if hasattr(self, "_detector"):
            self._detector.reset()
            self._initialization_cache = InitializationCache(
                self._meta_conf.adainit_cache_size,
                self._meta_conf.adainit_eps,
                self._meta_conf.adainit_signature_distance,
                self._meta_conf.adainit_cache_eviction,
                self._source_adaptable_state,
                self._meta_conf.adainit_fingerprint_magnitude_weight,
            )
            self._signature_window = RollingSignatureWindow(
                self._meta_conf.adainit_signature_window_size
            )
            self._selection_buffer = deque(
                maxlen=self._meta_conf.adainit_selection_window
            )
            self._selection_audit_labels = deque(
                maxlen=self._meta_conf.adainit_selection_window
            )
            self._knowledge_prediction_window = deque(
                maxlen=self._meta_conf.adainit_fingerprint_window
            )
            self._knowledge_state_snapshots = deque(maxlen=4)
            self._knowledge_state_snapshots.append(
                (0, clone_state_to_cpu(self._source_adaptable_state))
            )
            self._pending_pre_shift_state = None
            self._pending_hard_selection = None
            self._pending_sequential_selection = None
            self._stream_index = 0
            self._number_of_detections = 0
            self._candidate_selection_counts = {
                "current": 0,
                "source": 0,
                "history": 0,
                "native_anchor": 0,
            }
            self._oracle_lookahead_batches = []
            self._last_oracle_score_details = {}
            self._last_oracle_surrogate_details = {}
            self._last_causal_evidence_details = {}
            self._last_sequential_evidence_details = {}

    def set_oracle_lookahead(self, batches: List[Batch]) -> None:
        """Expose future batches only to the explicitly labeled oracle selector."""

        if (
            self._meta_conf.adainit_initialization_selector
            != "oracle_future_accuracy"
        ):
            raise RuntimeError(
                "Future batches may only be attached to oracle_future_accuracy."
            )
        horizon = self._meta_conf.adainit_oracle_horizon
        self._oracle_lookahead_batches = list(batches[:horizon])

    @staticmethod
    def _feature_matrix(features: torch.Tensor, channel_hint=None) -> torch.Tensor:
        if features.ndim == 3:
            # Transformer tokens: [B, N, D].
            return features.reshape(-1, features.shape[-1])
        if features.ndim == 4:
            # CNN maps are BCHW. Some timm PatchEmbed variants use BHWC.
            if channel_hint is not None and features.shape[-1] == channel_hint:
                return features.reshape(-1, features.shape[-1])
            return features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        raise RuntimeError(
            "AdaInit's fixed stem must return a 3-D token tensor or 4-D feature map; "
            f"got shape={tuple(features.shape)}."
        )

    def _extract_signature(self, inputs: torch.Tensor) -> DomainSignature:
        """Extract shallow statistics without touching adaptable layers."""

        with torch.no_grad():
            if hasattr(self._model, "patch_embed"):
                features = self._model.patch_embed(inputs)
                channel_hint = getattr(self._model.patch_embed, "embed_dim", None)
            elif hasattr(self._model, "conv1"):
                features = self._model.conv1(inputs)
                channel_hint = getattr(self._model.conv1, "out_channels", None)
            else:
                raise NotImplementedError(
                    "AdaInit currently supports models exposing patch_embed or conv1 "
                    "as a fixed shallow stem."
                )
            feature_matrix = self._feature_matrix(features, channel_hint).float()
            mean = feature_matrix.mean(dim=0)
            variance = feature_matrix.var(dim=0, unbiased=False)
        return DomainSignature(mean=mean, variance=variance).clone_cpu()

    @contextmanager
    def _fork_augmentation_rng(self, seed: int):
        """Make shared candidate views reproducible without leaking RNG state."""

        numpy_state = np.random.get_state()
        python_state = random.getstate()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            np.random.seed(seed % (2**32 - 1))
            random.seed(seed)
            try:
                yield
            finally:
                np.random.set_state(numpy_state)
                random.setstate(python_state)

    def _make_shared_views(self, batch: Batch):
        seed = self._meta_conf.seed + self._stream_index
        samples = list(self._selection_buffer)
        if not samples:
            samples = [batch._x[0].detach().cpu()]
        return self._make_views_for_samples(samples, seed)

    def _make_views_for_samples(self, samples, seed: int):
        """Create candidate-shared transformed views without changing RNG state."""

        transform_kwargs = {"data_name": self._meta_conf.base_data_name}
        with self._fork_augmentation_rng(seed):
            views = [
                [
                    self._transform_helper(
                        sample,
                        **transform_kwargs,
                    )
                    for _ in range(self._meta_conf.adainit_num_views)
                ]
                for sample in samples
            ]
        view_tensor = torch.stack(
            [torch.stack(sample_views) for sample_views in views]
        )
        # Keep the potentially large view bank on CPU. Candidate evaluation
        # transfers only one inference micro-batch at a time.
        return view_tensor.flatten(0, 1), len(samples)

    def _oracle_future_view_statistics(
        self,
        batch: Batch,
        stream_index: int,
    ) -> Dict[str, Any]:
        """Measure one branch's transformed-view evidence for diagnostics only.

        Probabilities are averaged across transformed views *within this one
        branch*. No prediction, score, or parameter is combined across candidate
        initializations, and these values never enter the Oracle hard decision.
        """

        shared_views, number_of_samples = self._make_views_for_samples(
            [sample.detach().cpu() for sample in batch._x],
            self._meta_conf.seed + stream_index,
        )
        with torch.no_grad(), fork_rng_with_seed(self._meta_conf.seed):
            logits = self._forward_candidate_views(shared_views)
        statistics = self._view_statistics(logits, number_of_samples)
        probabilities = torch.softmax(logits.float(), dim=1).reshape(
            number_of_samples,
            self._meta_conf.adainit_num_views,
            -1,
        )
        return {
            **statistics,
            "probability_sum": probabilities.mean(dim=1).sum(dim=0).detach().cpu(),
            "samples": number_of_samples,
        }

    def _forward_candidate_views(self, shared_views: torch.Tensor) -> torch.Tensor:
        micro_batch_size = self._meta_conf.adainit_candidate_eval_batch_size
        if micro_batch_size <= 0:
            return self._model(shared_views.to(self._meta_conf.device))
        return torch.cat(
            [
                self._model(chunk.to(self._meta_conf.device))
                for chunk in shared_views.split(micro_batch_size)
            ],
            dim=0,
        )

    def _counterfactual_batch(self, current_batch: Batch) -> Batch:
        if self._meta_conf.adainit_counterfactual_batch_source == "current":
            return current_batch
        samples = list(self._selection_buffer)
        if not samples:
            return current_batch
        inputs = torch.stack(samples).to(self._meta_conf.device)
        # Backends consume only x for their unsupervised update. Explicit dummy
        # labels prevent buffered ground truth from entering candidate selection.
        labels = torch.zeros(len(samples), dtype=torch.long, device=inputs.device)
        return Batch(inputs, labels)

    def _build_candidates(
        self,
        instantaneous_signature: DomainSignature,
        include_source: bool = True,
        trigger_reason: Optional[str] = None,
        current_candidate: Optional[InitializationCandidate] = None,
        query_knowledge_fingerprint: Optional[KnowledgeFingerprint] = None,
    ) -> List[InitializationCandidate]:
        current = (
            InitializationCandidate(
                "current",
                self._capture_adaptable_state(),
                optimizer_state=copy.deepcopy(self._optimizer.state_dict()),
                backend_runtime_state=copy.deepcopy(
                    self._backend_runtime_state
                ),
            )
            if current_candidate is None
            else InitializationCandidate(
                "current",
                clone_state_to_cpu(current_candidate.adaptable_state),
                optimizer_state=clone_state_to_cpu(
                    current_candidate.optimizer_state
                ),
                backend_runtime_state=clone_state_to_cpu(
                    current_candidate.backend_runtime_state
                ),
            )
        )
        source = InitializationCandidate("source", self._source_adaptable_state)
        native_anchor = (
            None
            if self._native_anchor_state is None
            else InitializationCandidate(
                "native_anchor",
                self._native_anchor_state,
                optimizer_state=copy.deepcopy(
                    self._native_anchor_optimizer.state_dict()
                ),
                backend_runtime_state=copy.deepcopy(
                    self._native_anchor_runtime_state
                ),
            )
        )
        fingerprint_retrieval = bool(
            self._meta_conf.adainit_history_retrieval
            == "knowledge_fingerprint"
        )
        if fingerprint_retrieval:
            ranked_fingerprints = (
                []
                if query_knowledge_fingerprint is None
                else self._initialization_cache.ranked_by_knowledge_fingerprint(
                    query_knowledge_fingerprint
                )
            )
            ranked_fingerprints = [
                item
                for item in ranked_fingerprints
                if self._stream_index - item[0].insertion_time
                >= self._meta_conf.adainit_history_min_age
            ]
            ranked_fingerprints = [
                item
                for item in ranked_fingerprints
                if not (
                    self._meta_conf.adainit_history_fingerprint_max_distance
                    >= 0
                    and item[1]
                    > self._meta_conf.adainit_history_fingerprint_max_distance
                    or self._meta_conf.adainit_history_fingerprint_max_ratio
                    >= 0
                    and item[2]
                    > self._meta_conf.adainit_history_fingerprint_max_ratio
                )
            ]
            maximum_histories = self._meta_conf.adainit_max_history_candidates
            if maximum_histories > 0:
                ranked_fingerprints = ranked_fingerprints[:maximum_histories]
            histories = [
                InitializationCandidate(
                    "history",
                    entry.adaptable_state,
                    optimizer_state=entry.optimizer_state,
                    backend_runtime_state=entry.backend_runtime_state,
                    cache_insertion_time=entry.insertion_time,
                    knowledge_distance=distance,
                    knowledge_distance_ratio=ratio,
                    cache_health_score=entry.health_score,
                )
                for entry, distance, ratio in ranked_fingerprints
            ]
        else:
            ranked_entries = self._initialization_cache.ranked_with_match(
                instantaneous_signature
            )
            ranked_entries = [
                item
                for item in ranked_entries
                if self._stream_index - item[0].insertion_time
                >= self._meta_conf.adainit_history_min_age
            ]
            maximum_histories = self._meta_conf.adainit_max_history_candidates
            if maximum_histories > 0:
                ranked_entries = ranked_entries[:maximum_histories]
            histories = [
                InitializationCandidate(
                    "history",
                    entry.adaptable_state,
                    distance,
                    ratio,
                    entry.optimizer_state,
                    entry.backend_runtime_state,
                    entry.insertion_time,
                    calibrated_ratio,
                    cache_health_score=entry.health_score,
                )
                for entry, distance, ratio, calibrated_ratio in ranked_entries
            ]

        mode = self._meta_conf.adainit_candidate_mode
        if mode == "full":
            if not fingerprint_retrieval:
                histories = [
                    history
                    for history in histories
                    if not (
                        (
                            self._meta_conf.adainit_history_match_ratio >= 0
                            and history.signature_distance_ratio
                            > self._meta_conf.adainit_history_match_ratio
                        )
                        or (
                            self._meta_conf.adainit_history_max_distance >= 0
                            and history.signature_distance
                            > self._meta_conf.adainit_history_max_distance
                        )
                        or (
                            self._meta_conf.adainit_history_radius_multiplier >= 0
                            and (
                                history.signature_calibrated_ratio is None
                                or history.signature_calibrated_ratio
                                > self._meta_conf.adainit_history_radius_multiplier
                            )
                        )
                    )
                ]
            use_source = bool(
                include_source
                and not (
                    histories
                    and self._meta_conf.adainit_source_for_unseen_only
                )
            )
            selected_candidates = (
                [current]
                + ([source] if use_source else [])
                + ([] if native_anchor is None else [native_anchor])
                + histories
            )
        elif mode == "current_only":
            selected_candidates = [current]
        elif mode == "source_only":
            selected_candidates = [source]
        elif mode == "nearest_history":
            selected_candidates = [current] if not histories else [histories[0]]
        elif mode == "counterfactual_history_only":
            selected_candidates = [current] if not histories else histories
        else:
            raise ValueError(f"Unsupported AdaInit candidate mode: {mode}")
        return self._share_candidate_runtime_states(
            selected_candidates, current.backend_runtime_state
        )

    def _build_oracle_candidates(
        self, instantaneous_signature: DomainSignature
    ) -> List[InitializationCandidate]:
        """Build an ungated upper-bound set from current, source, and the cache."""

        current = InitializationCandidate(
            "current",
            self._capture_adaptable_state(),
            optimizer_state=copy.deepcopy(self._optimizer.state_dict()),
            backend_runtime_state=copy.deepcopy(self._backend_runtime_state),
        )
        candidates = [current]
        if self._meta_conf.adainit_oracle_include_source:
            candidates.append(
                InitializationCandidate("source", self._source_adaptable_state)
            )
        if self._native_anchor_state is not None:
            candidates.append(
                InitializationCandidate(
                    "native_anchor",
                    self._native_anchor_state,
                    optimizer_state=copy.deepcopy(
                        self._native_anchor_optimizer.state_dict()
                    ),
                    backend_runtime_state=copy.deepcopy(
                        self._native_anchor_runtime_state
                    ),
                )
            )

        ranked_entries = self._initialization_cache.ranked_with_match(
            instantaneous_signature
        )
        maximum = self._meta_conf.adainit_oracle_max_history_candidates
        if maximum > 0:
            ranked_entries = ranked_entries[:maximum]
        for entry, distance, ratio, calibrated_ratio in ranked_entries:
            candidates.append(
                InitializationCandidate(
                    "history",
                    entry.adaptable_state,
                    signature_distance=distance,
                    signature_distance_ratio=ratio,
                    optimizer_state=entry.optimizer_state,
                    backend_runtime_state=entry.backend_runtime_state,
                    cache_insertion_time=entry.insertion_time,
                    signature_calibrated_ratio=calibrated_ratio,
                )
            )
        return candidates

    @staticmethod
    def _candidate_identifier(candidate: InitializationCandidate) -> str:
        if candidate.cache_insertion_time is None:
            return candidate.name
        return f"{candidate.name}@{candidate.cache_insertion_time}"

    def _branch_optimizer_and_runtime(
        self, candidate: InitializationCandidate
    ):
        optimizer = self._backend.build_optimizer(
            self._meta_conf, self._adapt_params
        )
        should_restore_optimizer = bool(
            candidate.optimizer_state is not None
            and (
                candidate.name == "current"
                and not self._meta_conf.adainit_reset_current_optimizer
                or candidate.name == "native_anchor"
                or candidate.name == "history"
                and self._meta_conf.adainit_cache_optimizer_state
            )
        )
        if should_restore_optimizer:
            optimizer.load_state_dict(copy.deepcopy(candidate.optimizer_state))
        if self._backend.share_runtime_state_across_candidates:
            runtime = (
                self._backend.fresh_runtime_state()
                if candidate.backend_runtime_state is None
                else copy.deepcopy(candidate.backend_runtime_state)
            )
        else:
            runtime = (
                self._backend.fresh_runtime_state()
                if candidate.backend_runtime_state is None
                or not should_restore_optimizer
                else copy.deepcopy(candidate.backend_runtime_state)
            )
        return optimizer, runtime

    def _share_candidate_runtime_states(
        self,
        candidates: List[InitializationCandidate],
        reference_runtime_state: Optional[Dict[str, Any]],
    ) -> List[InitializationCandidate]:
        """Give stateful-backend candidates an identical causal starting state."""

        if not self._backend.share_runtime_state_across_candidates:
            return candidates
        reference = (
            self._backend.fresh_runtime_state()
            if reference_runtime_state is None
            else reference_runtime_state
        )
        for candidate in candidates:
            candidate.backend_runtime_state = clone_state_to_cpu(reference)
        return candidates

    def _reset_after_hard_initialization(self) -> None:
        """Start a fresh optimizer while retaining only approved runtime state."""

        runtime_state = (
            copy.deepcopy(self._backend_runtime_state)
            if self._backend.preserve_runtime_state_on_hard_initialization
            else None
        )
        self._reset_backend_optimizer(runtime_state=runtime_state)

    def _oracle_future_accuracy(
        self,
        candidate: InitializationCandidate,
        current_batch: Batch,
        timer: Timer,
    ) -> Dict[str, float]:
        """Replay the actual future stream from one initialization.

        This function intentionally reads future labels.  It is a diagnostic
        upper bound, not an unsupervised selector and not a reportable TTA run.
        """

        self._load_adaptable_state(candidate.adaptable_state)
        optimizer, runtime = self._branch_optimizer_and_runtime(candidate)

        backend_losses = []

        def adapt(batch: Batch, number_of_steps: int):
            nonlocal optimizer, runtime
            for _ in range(number_of_steps):
                result = self._backend.adapt_step(
                    self._model,
                    optimizer,
                    batch,
                    runtime,
                    timer,
                    random_seed=self._meta_conf.seed,
                )
                diagnostic_loss = (
                    result.loss
                    if result.surrogate_loss is None
                    else result.surrogate_loss
                )
                if np.isfinite(diagnostic_loss):
                    backend_losses.append(float(diagnostic_loss))
                if result.reset_requested:
                    self._load_adaptable_state(self._source_adaptable_state)
                    optimizer = self._backend.build_optimizer(
                        self._meta_conf, self._adapt_params
                    )
                    # Native SAR keeps the triggering recovery EMA after reset.
                    runtime = runtime

        adapt(
            current_batch,
            self._get_adaptation_steps(index=self._stream_index),
        )
        future_view_statistics = []
        if self._meta_conf.adainit_oracle_log_future_view_surrogates:
            future_view_statistics.append(
                self._oracle_future_view_statistics(
                    current_batch,
                    self._stream_index,
                )
            )
        correct = 0
        total = 0
        predictive_entropy_sum = 0.0
        probability_sum = None
        with timer("adainit.oracle_future_replay"):
            for offset, future_batch in enumerate(
                self._oracle_lookahead_batches, start=1
            ):
                with torch.no_grad(), fork_rng_with_seed(self._meta_conf.seed):
                    prediction = self._model(future_batch._x)
                    probabilities = torch.softmax(prediction.float(), dim=1)
                    predictive_entropy_sum += float(
                        (-(probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()).sum(dim=1))
                        .sum()
                        .item()
                    )
                    batch_probability_sum = probabilities.sum(dim=0).detach().cpu()
                    probability_sum = (
                        batch_probability_sum
                        if probability_sum is None
                        else probability_sum + batch_probability_sum
                    )
                correct += int(
                    prediction.argmax(dim=1)
                    .eq(future_batch._y)
                    .sum()
                    .detach()
                    .item()
                )
                total += len(future_batch)
                adapt(
                    future_batch,
                    self._get_adaptation_steps(index=self._stream_index + offset),
                )
                if self._meta_conf.adainit_oracle_log_future_view_surrogates:
                    future_view_statistics.append(
                        self._oracle_future_view_statistics(
                            future_batch,
                            self._stream_index + offset,
                        )
                    )
        mean_predictive_entropy = predictive_entropy_sum / max(total, 1)
        if probability_sum is None:
            marginal_predictive_entropy = float("nan")
        else:
            marginal_probabilities = (probability_sum / max(total, 1)).clamp_min(
                1e-12
            )
            marginal_predictive_entropy = float(
                (-(marginal_probabilities * marginal_probabilities.log()).sum()).item()
            )
        result = {
            "accuracy": 100.0 * correct / max(total, 1),
            "correct": float(correct),
            "samples": float(total),
            "mean_predictive_entropy": mean_predictive_entropy,
            "marginal_predictive_entropy": marginal_predictive_entropy,
            "information_maximization_score": (
                mean_predictive_entropy - marginal_predictive_entropy
            ),
            "mean_backend_loss": (
                float(np.mean(backend_losses))
                if backend_losses
                else float("inf")
            ),
        }
        if future_view_statistics:
            evidence_samples = sum(
                item["samples"] for item in future_view_statistics
            )
            evidence_probability_sum = sum(
                (item["probability_sum"] for item in future_view_statistics),
                torch.zeros_like(future_view_statistics[0]["probability_sum"]),
            )
            evidence_marginal = (
                evidence_probability_sum / max(evidence_samples, 1)
            ).clamp_min(1e-12)
            evidence_context_entropy = float(
                (-(evidence_marginal * evidence_marginal.log()).sum()).item()
            )
            mean_marginal_entropy = float(
                np.average(
                    [item["marginal_entropy"] for item in future_view_statistics],
                    weights=[item["samples"] for item in future_view_statistics],
                )
            )
            result.update(
                {
                    "evidence_samples": float(evidence_samples),
                    "evidence_mean_marginal_entropy": mean_marginal_entropy,
                    "evidence_mean_view_entropy": float(
                        np.average(
                            [
                                item["mean_view_entropy"]
                                for item in future_view_statistics
                            ],
                            weights=[
                                item["samples"] for item in future_view_statistics
                            ],
                        )
                    ),
                    "evidence_mean_view_jsd": float(
                        np.average(
                            [item["view_jsd"] for item in future_view_statistics],
                            weights=[
                                item["samples"] for item in future_view_statistics
                            ],
                        )
                    ),
                    "evidence_context_entropy": evidence_context_entropy,
                    "evidence_information_maximization_score": (
                        mean_marginal_entropy - evidence_context_entropy
                    ),
                }
            )
        return result

    def _select_initialization_oracle(
        self,
        batch: Batch,
        instantaneous_signature: DomainSignature,
        timer: Timer,
    ) -> InitializationCandidate:
        """Choose the candidate with the best true future-stream Acc@H."""

        candidates = self._build_oracle_candidates(instantaneous_signature)
        if not self._oracle_lookahead_batches:
            self._last_candidate_scores = {}
            self._last_candidate_score_details = {}
            self._last_oracle_score_details = {}
            self._last_oracle_surrogate_details = {}
            return candidates[0]

        original_state = self._capture_adaptable_state()
        self._last_oracle_score_details = {}
        self._last_oracle_surrogate_details = {}
        scored_candidates = []
        if self._meta_conf.adainit_oracle_log_surrogates:
            shared_views, number_of_selection_samples = self._make_shared_views(batch)
            counterfactual_batch = self._counterfactual_batch(batch)
        else:
            shared_views = None
            number_of_selection_samples = 0
            counterfactual_batch = None
        try:
            for candidate in candidates:
                identifier = self._candidate_identifier(candidate)
                if self._meta_conf.adainit_oracle_log_surrogates:
                    self._last_oracle_surrogate_details[identifier] = (
                        self._score_counterfactual_candidate(
                            candidate,
                            counterfactual_batch,
                            shared_views,
                            number_of_selection_samples,
                            timer,
                        )
                    )
                    self._load_adaptable_state(original_state)
                details = self._oracle_future_accuracy(
                    candidate, batch, timer
                )
                self._last_oracle_score_details[identifier] = details
                scored_candidates.append((details["accuracy"], candidate))
                self._load_adaptable_state(original_state)
        finally:
            self._load_adaptable_state(original_state)

        current_score, current_candidate = next(
            item for item in scored_candidates if item[1].name == "current"
        )
        best_score, best_candidate = max(
            scored_candidates, key=lambda item: item[0]
        )
        margin = self._meta_conf.adainit_oracle_min_accuracy_gain
        if best_candidate.name != "current" and best_score <= current_score + margin:
            best_candidate = current_candidate
        self._last_candidate_scores = {}
        self._last_candidate_score_details = {}
        return best_candidate

    def _view_statistics(
        self, logits: torch.Tensor, number_of_samples: int
    ) -> Dict[str, float]:
        probabilities = torch.softmax(logits.float(), dim=1).reshape(
            number_of_samples,
            self._meta_conf.adainit_num_views,
            -1,
        )
        marginal = probabilities.mean(dim=1).clamp_min(1e-12)
        marginal_entropy = float(
            (-(marginal * marginal.log()).sum(dim=1)).mean().item()
        )
        view_probabilities = probabilities.clamp_min(1e-12)
        mean_view_entropy = float(
            (-(view_probabilities * view_probabilities.log()).sum(dim=2))
            .mean()
            .item()
        )
        context_marginal = marginal.mean(dim=0).clamp_min(1e-12)
        context_entropy = float(
            (-(context_marginal * context_marginal.log()).sum()).item()
        )
        context_diversity = max(0.0, context_entropy - marginal_entropy)
        return {
            "marginal_entropy": marginal_entropy,
            "mean_view_entropy": mean_view_entropy,
            "view_jsd": marginal_entropy - mean_view_entropy,
            "context_diversity": context_diversity,
        }

    def _audit_view_ensemble_accuracy(
        self, logits: torch.Tensor, number_of_samples: int
    ) -> Optional[float]:
        """Return a label-based audit metric that cannot affect selection."""

        if not self._meta_conf.adainit_audit_candidate_accuracy:
            return None
        labels = list(self._selection_audit_labels)
        if len(labels) != number_of_samples:
            raise RuntimeError(
                "AdaInit candidate audit buffer mismatch: "
                f"labels={len(labels)}, samples={number_of_samples}."
            )
        probabilities = torch.softmax(logits.float(), dim=1).reshape(
            number_of_samples,
            self._meta_conf.adainit_num_views,
            -1,
        )
        predictions = probabilities.mean(dim=1).argmax(dim=1)
        targets = torch.stack(labels).reshape(-1).to(predictions.device)
        if targets.numel() != number_of_samples:
            raise RuntimeError(
                "AdaInit candidate audit expects one target per buffered sample; "
                f"got {targets.numel()} for {number_of_samples}."
            )
        return float(predictions.eq(targets).float().mean().mul(100.0).item())

    def _score_counterfactual_candidate(
        self,
        candidate: InitializationCandidate,
        batch: Batch,
        shared_views: torch.Tensor,
        number_of_selection_samples: int,
        timer: Timer,
    ) -> Dict[str, float]:
        self._load_adaptable_state(candidate.adaptable_state)
        entropy_before = None
        if self._meta_conf.adainit_recovery_score == "entropy_delta":
            with timer("adainit.candidate_view_evaluation"):
                with torch.no_grad(), fork_rng_with_seed(self._meta_conf.seed):
                    logits_before = self._forward_candidate_views(shared_views)
            entropy_before = self._view_statistics(
                logits_before, number_of_selection_samples
            )[
                "marginal_entropy"
            ]

        temporary_optimizer = self._backend.build_optimizer(
            self._meta_conf, self._adapt_params
        )
        if (
            (
                self._meta_conf.adainit_cache_optimizer_state
                or candidate.name == "native_anchor"
            )
            and candidate.optimizer_state is not None
        ):
            temporary_optimizer.load_state_dict(
                copy.deepcopy(candidate.optimizer_state)
            )
        temporary_runtime = (
            self._backend.fresh_runtime_state()
            if candidate.backend_runtime_state is None
            else copy.deepcopy(candidate.backend_runtime_state)
        )
        backend_losses = []
        for _ in range(self._meta_conf.adainit_counterfactual_steps):
            result = self._backend.adapt_step(
                self._model,
                temporary_optimizer,
                batch,
                temporary_runtime,
                timer,
                random_seed=self._meta_conf.seed,
            )
            candidate_backend_loss = (
                result.loss
                if result.surrogate_loss is None
                else result.surrogate_loss
            )
            if np.isfinite(candidate_backend_loss):
                backend_losses.append(candidate_backend_loss)
            if result.reset_requested:
                self._load_adaptable_state(self._source_adaptable_state)
                temporary_optimizer = self._backend.build_optimizer(
                    self._meta_conf, self._adapt_params
                )
                temporary_runtime = self._backend.fresh_runtime_state()
        backend_loss = (
            float(np.mean(backend_losses)) if backend_losses else float("inf")
        )

        with timer("adainit.candidate_view_evaluation"):
            with torch.no_grad(), fork_rng_with_seed(self._meta_conf.seed):
                view_logits = self._forward_candidate_views(shared_views)
        post_statistics = self._view_statistics(
            view_logits, number_of_selection_samples
        )
        # Audit only: computed after the unlabeled recovery statistics and never
        # referenced by the score or feasibility rules below.
        audit_accuracy = self._audit_view_ensemble_accuracy(
            view_logits, number_of_selection_samples
        )
        entropy_after = post_statistics["marginal_entropy"]
        if self._meta_conf.adainit_recovery_score == "post_entropy":
            score = entropy_after
        elif self._meta_conf.adainit_recovery_score == "entropy_delta":
            score = entropy_after - entropy_before
        elif self._meta_conf.adainit_recovery_score == "backend_loss":
            score = backend_loss
        else:
            score = (
                post_statistics["view_jsd"]
                - self._meta_conf.adainit_context_diversity_weight
                * post_statistics["context_diversity"]
            )
        return {
            "score": score,
            "backend_loss": backend_loss,
            "entropy_before": entropy_before,
            "entropy_after": entropy_after,
            "mean_view_entropy": post_statistics["mean_view_entropy"],
            "view_jsd": post_statistics["view_jsd"],
            "context_diversity": post_statistics["context_diversity"],
            "audit_view_ensemble_accuracy": audit_accuracy,
        }

    def _select_initialization(
        self,
        batch: Batch,
        instantaneous_signature: DomainSignature,
        timer: Timer,
        include_source: bool = True,
        trigger_reason: Optional[str] = None,
        drift_strength_ratio: Optional[float] = None,
    ) -> InitializationCandidate:
        candidates = self._build_candidates(
            instantaneous_signature,
            include_source=include_source,
            trigger_reason=trigger_reason,
        )
        mode = self._meta_conf.adainit_candidate_mode

        matched_history = next(
            (candidate for candidate in candidates if candidate.name == "history"),
            None,
        )
        if (
            mode == "full"
            and self._meta_conf.adainit_prefer_matched_history
            and matched_history is not None
            and self._meta_conf.adainit_history_match_ratio >= 0
        ):
            self._last_candidate_scores = {}
            self._last_candidate_score_details = {}
            return matched_history

        # Fixed-candidate and nearest-history ablations do not need a temporary
        # update. The complete method and history-only counterfactual ablation do.
        if len(candidates) == 1 and mode != "counterfactual_history_only":
            self._last_candidate_scores = {}
            self._last_candidate_score_details = {}
            return candidates[0]

        shared_views, number_of_selection_samples = self._make_shared_views(batch)
        counterfactual_batch = self._counterfactual_batch(batch)
        scored_candidates = []
        self._last_candidate_score_details = {}
        for candidate in candidates:
            identifier = self._candidate_identifier(candidate)
            details = self._score_counterfactual_candidate(
                candidate,
                counterfactual_batch,
                shared_views,
                number_of_selection_samples,
                timer,
            )
            effective_score = details["score"]
            if (
                candidate.name == "history"
                and candidate.signature_distance_ratio is not None
            ):
                effective_score -= (
                    self._meta_conf.adainit_history_score_bonus
                    * max(
                        0.0,
                        self._meta_conf.adainit_history_match_ratio
                        - candidate.signature_distance_ratio,
                    )
                )
            details["raw_score"] = details["score"]
            details["score"] = effective_score
            scored_candidates.append((effective_score, candidate))
            self._last_candidate_score_details[identifier] = details
            self._last_candidate_score_details[identifier][
                "signature_distance"
            ] = candidate.signature_distance
            self._last_candidate_score_details[identifier][
                "signature_distance_ratio"
            ] = candidate.signature_distance_ratio
            self._last_candidate_score_details[identifier][
                "signature_calibrated_ratio"
            ] = candidate.signature_calibrated_ratio
            self._last_candidate_score_details[identifier][
                "cache_insertion_time"
            ] = candidate.cache_insertion_time
        self._last_candidate_scores = {
            self._candidate_identifier(candidate): score
            for score, candidate in scored_candidates
        }
        # The exact paper rule is recovered with margin=0. A positive margin is
        # a conservative ablation for noisy single-sample counterfactual scores:
        # switching away from the current trajectory must improve the surrogate
        # by at least the configured amount.
        current_item = next(
            (
                (score, candidate)
                for score, candidate in scored_candidates
                if candidate.name == "current"
            ),
            None,
        )
        if current_item is None:
            return min(scored_candidates, key=lambda item: item[0])[1]

        history_entropy_limit = (
            self._meta_conf.adainit_periodic_history_max_entropy_increase
            if trigger_reason == "periodic"
            else self._meta_conf.adainit_history_max_entropy_increase
        )
        current_entropy = self._last_candidate_score_details["current"][
            "entropy_after"
        ]
        # Treat the safety rules as feasibility constraints. If the numerically
        # best candidate is rejected, continue to the next candidate instead of
        # prematurely falling back to current and discarding a valid alternative.
        for candidate_score, candidate in sorted(
            scored_candidates, key=lambda item: item[0]
        ):
            if candidate.name == "current":
                return candidate
            identifier = self._candidate_identifier(candidate)
            candidate_entropy = self._last_candidate_score_details[identifier][
                "entropy_after"
            ]
            if (
                candidate.name == "source"
                and self._meta_conf.adainit_source_min_current_entropy >= 0
                and current_entropy
                < self._meta_conf.adainit_source_min_current_entropy
            ):
                continue
            if (
                candidate.name == "history"
                and self._meta_conf.adainit_history_min_drift_ratio > 0
                and (
                    drift_strength_ratio is None
                    or drift_strength_ratio
                    < self._meta_conf.adainit_history_min_drift_ratio
                )
            ):
                continue
            if (
                candidate.name == "history"
                and history_entropy_limit >= 0
                and candidate_entropy > current_entropy + history_entropy_limit
            ):
                continue
            if candidate.name == "source":
                margin = self._meta_conf.adainit_source_selection_margin
            elif candidate.name == "native_anchor":
                margin = self._meta_conf.adainit_native_anchor_selection_margin
            elif trigger_reason == "periodic":
                margin = self._meta_conf.adainit_periodic_selection_margin
            else:
                margin = self._meta_conf.adainit_selection_margin
            if candidate_score > current_item[0] - margin:
                continue
            if (
                self._meta_conf.adainit_max_marginal_entropy_increase >= 0
                and candidate_entropy
                > current_entropy
                + self._meta_conf.adainit_max_marginal_entropy_increase
            ):
                continue
            return candidate
        return current_item[1]

    def _start_causal_hard_selection(
        self,
        instantaneous_signature: DomainSignature,
        trigger_reason: str,
        drift_strength_ratio: Optional[float],
        include_source: bool,
    ) -> bool:
        """Freeze hard candidate branches for a causal multi-frame decision."""

        if self._pending_hard_selection is not None:
            return False
        formal_state = self._capture_adaptable_state()
        candidates = self._build_candidates(
            instantaneous_signature,
            include_source=include_source,
            trigger_reason=trigger_reason,
        )
        if (
            self._meta_conf.adainit_history_min_drift_ratio > 0
            and (
                drift_strength_ratio is None
                or drift_strength_ratio
                < self._meta_conf.adainit_history_min_drift_ratio
            )
        ):
            candidates = [
                candidate
                for candidate in candidates
                if candidate.name != "history"
            ]

        branches = []
        try:
            for candidate in candidates:
                self._load_adaptable_state(candidate.adaptable_state)
                optimizer, runtime = self._branch_optimizer_and_runtime(candidate)
                branches.append(
                    CausalEvidenceBranch(
                        candidate=InitializationCandidate(
                            name=candidate.name,
                            adaptable_state=clone_state_to_cpu(
                                candidate.adaptable_state
                            ),
                            signature_distance=candidate.signature_distance,
                            signature_distance_ratio=(
                                candidate.signature_distance_ratio
                            ),
                            cache_insertion_time=candidate.cache_insertion_time,
                            signature_calibrated_ratio=(
                                candidate.signature_calibrated_ratio
                            ),
                        ),
                        optimizer_state=clone_state_to_cpu(optimizer.state_dict()),
                        backend_runtime_state=clone_state_to_cpu(runtime),
                    )
                )
        finally:
            self._load_adaptable_state(formal_state)
        self._pending_hard_selection = PendingHardSelection(
            trigger_index=self._stream_index,
            trigger_reason=trigger_reason,
            drift_strength_ratio=drift_strength_ratio,
            branches=branches,
        )
        return True

    def _advance_causal_hard_selection(
        self,
        batch: Batch,
        number_of_steps: int,
        timer: Timer,
    ) -> Optional[CausalEvidenceBranch]:
        """Advance independent branches and hard-commit one after the horizon.

        Each branch receives the same causal singleton stream and transformed
        views. Candidate predictions and parameters are never combined. Until a
        decision is committed, the uninterrupted formal trajectory remains the
        only trajectory used for benchmark predictions.
        """

        pending = self._pending_hard_selection
        if pending is None:
            return None
        formal_state = self._capture_adaptable_state()
        shared_views, number_of_samples = self._make_views_for_samples(
            [batch._x[0].detach().cpu()],
            self._meta_conf.seed + self._stream_index,
        )
        try:
            for branch in pending.branches:
                self._load_adaptable_state(branch.candidate.adaptable_state)
                optimizer = self._backend.build_optimizer(
                    self._meta_conf, self._adapt_params
                )
                optimizer.load_state_dict(copy.deepcopy(branch.optimizer_state))
                runtime = copy.deepcopy(branch.backend_runtime_state)
                should_score = (
                    pending.processed_samples
                    >= self._meta_conf.adainit_evidence_warmup
                )
                if (
                    should_score
                    and self._meta_conf.adainit_evidence_timing
                    == "prequential"
                ):
                    with timer("adainit.causal_evidence_view_evaluation"):
                        with torch.no_grad(), fork_rng_with_seed(
                            self._meta_conf.seed
                        ):
                            logits = self._forward_candidate_views(shared_views)
                    statistics = self._view_statistics(
                        logits, number_of_samples
                    )
                    branch.entropy_sum += statistics["marginal_entropy"]
                    branch.view_jsd_sum += statistics["view_jsd"]
                    branch.observations += number_of_samples
                for _ in range(number_of_steps):
                    result = self._backend.adapt_step(
                        self._model,
                        optimizer,
                        batch,
                        runtime,
                        timer,
                        random_seed=self._meta_conf.seed,
                    )
                    if result.reset_requested:
                        self._load_adaptable_state(self._source_adaptable_state)
                        optimizer = self._backend.build_optimizer(
                            self._meta_conf, self._adapt_params
                        )
                branch.candidate.adaptable_state = self._capture_adaptable_state()
                branch.optimizer_state = clone_state_to_cpu(optimizer.state_dict())
                branch.backend_runtime_state = clone_state_to_cpu(runtime)
                if (
                    should_score
                    and self._meta_conf.adainit_evidence_timing == "post_update"
                ):
                    with timer("adainit.causal_evidence_view_evaluation"):
                        with torch.no_grad(), fork_rng_with_seed(
                            self._meta_conf.seed
                        ):
                            logits = self._forward_candidate_views(shared_views)
                    statistics = self._view_statistics(
                        logits, number_of_samples
                    )
                    branch.entropy_sum += statistics["marginal_entropy"]
                    branch.view_jsd_sum += statistics["view_jsd"]
                    branch.observations += number_of_samples
        finally:
            self._load_adaptable_state(formal_state)

        if not pending.branches:
            self._pending_hard_selection = None
            return None
        pending.processed_samples += number_of_samples
        if pending.processed_samples < self._meta_conf.adainit_evidence_horizon:
            return None
        selected = self._select_causal_evidence_branch(pending)
        self._pending_hard_selection = None
        return selected

    def _select_causal_evidence_branch(
        self,
        pending: PendingHardSelection,
    ) -> CausalEvidenceBranch:
        """Apply safety margins, then return exactly one complete branch."""

        scored = []
        self._last_causal_evidence_details = {}
        for branch in pending.branches:
            identifier = self._candidate_identifier(branch.candidate)
            mean_entropy = branch.entropy_sum / max(branch.observations, 1)
            effective_score = mean_entropy
            if (
                branch.candidate.name == "history"
                and branch.candidate.signature_distance_ratio is not None
            ):
                effective_score -= (
                    self._meta_conf.adainit_history_score_bonus
                    * max(
                        0.0,
                        self._meta_conf.adainit_history_match_ratio
                        - branch.candidate.signature_distance_ratio,
                    )
                )
            self._last_causal_evidence_details[identifier] = {
                "score": effective_score,
                "mean_marginal_entropy": mean_entropy,
                "mean_view_jsd": (
                    branch.view_jsd_sum / max(branch.observations, 1)
                ),
                "observations": branch.observations,
            }
            scored.append((effective_score, branch))

        current_score, current_branch = next(
            item for item in scored if item[1].candidate.name == "current"
        )
        current_entropy = self._last_causal_evidence_details["current"][
            "mean_marginal_entropy"
        ]
        history_entropy_limit = (
            self._meta_conf.adainit_periodic_history_max_entropy_increase
            if pending.trigger_reason == "periodic"
            else self._meta_conf.adainit_history_max_entropy_increase
        )
        for candidate_score, branch in sorted(scored, key=lambda item: item[0]):
            candidate = branch.candidate
            if candidate.name == "current":
                return branch
            identifier = self._candidate_identifier(candidate)
            candidate_entropy = self._last_causal_evidence_details[identifier][
                "mean_marginal_entropy"
            ]
            if (
                candidate.name == "source"
                and self._meta_conf.adainit_source_min_current_entropy >= 0
                and current_entropy
                < self._meta_conf.adainit_source_min_current_entropy
            ):
                continue
            if (
                candidate.name == "history"
                and history_entropy_limit >= 0
                and candidate_entropy > current_entropy + history_entropy_limit
            ):
                continue
            if candidate.name == "source":
                margin = self._meta_conf.adainit_source_selection_margin
            elif candidate.name == "native_anchor":
                margin = self._meta_conf.adainit_native_anchor_selection_margin
            elif pending.trigger_reason == "periodic":
                margin = self._meta_conf.adainit_periodic_selection_margin
            else:
                margin = self._meta_conf.adainit_selection_margin
            if candidate_score > current_score - margin:
                continue
            if (
                self._meta_conf.adainit_max_marginal_entropy_increase >= 0
                and candidate_entropy
                > current_entropy
                + self._meta_conf.adainit_max_marginal_entropy_increase
            ):
                continue
            return branch
        return current_branch

    def _commit_causal_evidence_branch(
        self,
        branch: CausalEvidenceBranch,
    ) -> None:
        """Install one branch model state; never merge it with another state."""

        if branch.candidate.name != "current":
            self._load_adaptable_state(branch.candidate.adaptable_state)
            # AdaInit chooses model initialization only. The selected non-current
            # model starts a fresh formal optimizer; branch optimizer state was
            # used solely to evaluate that initialization's recovery trajectory.
            self._reset_after_hard_initialization()
        self._candidate_selection_counts[branch.candidate.name] += 1

    def _start_sequential_view_selection(
        self,
        trigger_reason: str,
        drift_strength_ratio: Optional[float],
        include_source: bool,
    ) -> bool:
        """Start an early recovery proposal without changing the formal model."""

        if self._pending_sequential_selection is not None:
            return False
        current = InitializationCandidate(
            "current",
            self._capture_adaptable_state(),
            optimizer_state=clone_state_to_cpu(self._optimizer.state_dict()),
            backend_runtime_state=clone_state_to_cpu(
                self._backend_runtime_state
            ),
        )
        self._pending_sequential_selection = PendingSequentialSelection(
            trigger_index=self._stream_index,
            trigger_reason=trigger_reason,
            drift_strength_ratio=drift_strength_ratio,
            trigger_current=current,
            proposal_signature=None,
            include_source=include_source,
            buffered_inputs=[],
            branches=[],
        )
        return True

    def _update_provisional_signature(
        self,
        previous: Optional[DomainSignature],
        current: DomainSignature,
    ) -> DomainSignature:
        """Use the same momentum definition for the temporary and main MDS."""

        if previous is None:
            return current.clone_cpu()
        alpha = self._meta_conf.adainit_signature_momentum
        return DomainSignature(
            mean=(alpha * previous.mean + (1.0 - alpha) * current.mean),
            variance=(
                alpha * previous.variance
                + (1.0 - alpha) * current.variance
            ),
        ).clone_cpu()

    def _initialize_sequential_branches(
        self, pending: PendingSequentialSelection
    ) -> None:
        if pending.proposal_signature is None:
            return
        formal_state = self._capture_adaptable_state()
        query_knowledge_fingerprint = None
        if (
            self._meta_conf.adainit_history_retrieval
            == "knowledge_fingerprint"
        ):
            fingerprint_start = (
                self._source_adaptable_state
                if self._meta_conf.adainit_knowledge_fingerprint
                == "source_delta"
                else pending.trigger_current.adaptable_state
            )
            query_knowledge_fingerprint = build_knowledge_fingerprint(
                fingerprint_start,
                formal_state,
            )
        candidates = self._build_candidates(
            pending.proposal_signature,
            include_source=pending.include_source,
            trigger_reason=pending.trigger_reason,
            current_candidate=pending.trigger_current,
            query_knowledge_fingerprint=query_knowledge_fingerprint,
        )
        if (
            self._meta_conf.adainit_history_min_drift_ratio > 0
            and (
                pending.drift_strength_ratio is None
                or pending.drift_strength_ratio
                < self._meta_conf.adainit_history_min_drift_ratio
            )
        ):
            candidates = [
                candidate
                for candidate in candidates
                if candidate.name != "history"
            ]
        try:
            for candidate in candidates:
                self._load_adaptable_state(candidate.adaptable_state)
                optimizer, runtime = self._branch_optimizer_and_runtime(candidate)
                pending.branches.append(
                    SequentialEvidenceBranch(
                        candidate=InitializationCandidate(
                            name=candidate.name,
                            adaptable_state=clone_state_to_cpu(
                                candidate.adaptable_state
                            ),
                            signature_distance=candidate.signature_distance,
                            signature_distance_ratio=(
                                candidate.signature_distance_ratio
                            ),
                            cache_insertion_time=candidate.cache_insertion_time,
                            signature_calibrated_ratio=(
                                candidate.signature_calibrated_ratio
                            ),
                            knowledge_distance=candidate.knowledge_distance,
                            knowledge_distance_ratio=(
                                candidate.knowledge_distance_ratio
                            ),
                            cache_health_score=candidate.cache_health_score,
                        ),
                        optimizer_state=clone_state_to_cpu(
                            optimizer.state_dict()
                        ),
                        backend_runtime_state=clone_state_to_cpu(runtime),
                        marginal_entropies=deque(
                            maxlen=self._meta_conf.adainit_evidence_horizon
                        ),
                        view_jsds=deque(
                            maxlen=self._meta_conf.adainit_evidence_horizon
                        ),
                        marginal_probabilities=deque(
                            maxlen=self._meta_conf.adainit_evidence_horizon
                        ),
                    )
                )
        finally:
            self._load_adaptable_state(formal_state)

    def _process_sequential_inputs(
        self,
        pending: PendingSequentialSelection,
        inputs: List[torch.Tensor],
        first_observation_index: int,
        number_of_steps: int,
        timer: Timer,
    ) -> None:
        """Advance every hard branch on identical singleton inputs and views."""

        if not pending.branches:
            return
        formal_state = self._capture_adaptable_state()
        try:
            for offset, cpu_input in enumerate(inputs):
                observation_index = first_observation_index + offset
                view_seed = self._meta_conf.seed + pending.trigger_index + observation_index
                shared_views, number_of_samples = self._make_views_for_samples(
                    [cpu_input], view_seed
                )
                singleton_input = cpu_input.unsqueeze(0).to(
                    self._meta_conf.device
                )
                singleton = Batch(
                    singleton_input,
                    torch.zeros(
                        1,
                        dtype=torch.long,
                        device=singleton_input.device,
                    ),
                )
                for branch in pending.branches:
                    self._load_adaptable_state(
                        branch.candidate.adaptable_state
                    )
                    optimizer = self._backend.build_optimizer(
                        self._meta_conf, self._adapt_params
                    )
                    optimizer.load_state_dict(
                        copy.deepcopy(branch.optimizer_state)
                    )
                    runtime = copy.deepcopy(branch.backend_runtime_state)

                    def record_view_evidence() -> None:
                        with timer("adainit.sequential_view_evaluation"):
                            with torch.no_grad(), fork_rng_with_seed(view_seed):
                                logits = self._forward_candidate_views(
                                    shared_views
                                )
                        statistics = self._view_statistics(
                            logits, number_of_samples
                        )
                        branch.marginal_entropies.append(
                            statistics["marginal_entropy"]
                        )
                        branch.view_jsds.append(statistics["view_jsd"])
                        probabilities = torch.softmax(
                            logits.float(), dim=1
                        ).reshape(
                            number_of_samples,
                            self._meta_conf.adainit_num_views,
                            -1,
                        )
                        for marginal_probability in probabilities.mean(dim=1):
                            branch.marginal_probabilities.append(
                                marginal_probability.detach().cpu()
                            )
                        branch.observations += number_of_samples

                    if self._meta_conf.adainit_evidence_timing == "prequential":
                        record_view_evidence()
                    for _ in range(number_of_steps):
                        result = self._backend.adapt_step(
                            self._model,
                            optimizer,
                            singleton,
                            runtime,
                            timer,
                            random_seed=self._meta_conf.seed,
                        )
                        if result.reset_requested:
                            self._load_adaptable_state(
                                self._source_adaptable_state
                            )
                            optimizer = self._backend.build_optimizer(
                                self._meta_conf, self._adapt_params
                            )
                    branch.candidate.adaptable_state = (
                        self._capture_adaptable_state()
                    )
                    branch.optimizer_state = clone_state_to_cpu(
                        optimizer.state_dict()
                    )
                    branch.backend_runtime_state = clone_state_to_cpu(runtime)
                    if self._meta_conf.adainit_evidence_timing == "post_update":
                        record_view_evidence()
        finally:
            self._load_adaptable_state(formal_state)

    def _sequential_candidate_margin(
        self,
        candidate: InitializationCandidate,
        trigger_reason: str,
    ) -> float:
        if candidate.name == "source":
            return self._meta_conf.adainit_source_selection_margin
        if candidate.name == "native_anchor":
            return self._meta_conf.adainit_native_anchor_selection_margin
        if trigger_reason == "periodic":
            return self._meta_conf.adainit_periodic_selection_margin
        return self._meta_conf.adainit_selection_margin

    def _sequential_branch_score(
        self, branch: SequentialEvidenceBranch
    ) -> Dict[str, Any]:
        """Compute one query-conditioned hard-branch score on a rolling window."""

        entropies = np.asarray(
            list(branch.marginal_entropies), dtype=np.float64
        )
        view_jsds = np.asarray(list(branch.view_jsds), dtype=np.float64)
        mean_entropy = float(entropies.mean())
        mean_view_jsd = float(view_jsds.mean())
        context_entropy = 0.0
        if branch.marginal_probabilities:
            context_marginal = torch.stack(
                list(branch.marginal_probabilities)
            ).float().mean(dim=0).clamp_min(1e-12)
            context_entropy = float(
                (-(context_marginal * context_marginal.log()).sum()).item()
            )
        information_score = max(0.0, context_entropy - mean_entropy)
        if self._meta_conf.adainit_sequential_score == "view_infomax":
            base_values = (
                entropies
                + self._meta_conf.adainit_sequential_view_jsd_weight
                * view_jsds
            )
            score = float(base_values.mean()) - (
                self._meta_conf.adainit_sequential_context_weight
                * context_entropy
            )
        else:
            base_values = entropies
            score = mean_entropy
        return {
            "base_values": base_values,
            "score": score,
            "mean_marginal_entropy": mean_entropy,
            "mean_view_jsd": mean_view_jsd,
            "context_entropy": context_entropy,
            "information_score": information_score,
            "observations": int(entropies.size),
        }

    def _select_sequential_evidence_branch(
        self,
        pending: PendingSequentialSelection,
    ) -> Optional[SequentialEvidenceBranch]:
        """Return a non-current branch only after a positive advantage LCB."""

        if not pending.branches:
            return None
        current_branch = next(
            branch
            for branch in pending.branches
            if branch.candidate.name == "current"
        )
        current_statistics = self._sequential_branch_score(current_branch)
        current_values = current_statistics["base_values"]
        if (
            current_values.size
            < self._meta_conf.adainit_evidence_min_samples
        ):
            return None

        self._last_sequential_evidence_details = {}
        eligible = []
        current_mean = current_statistics["mean_marginal_entropy"]
        for branch in pending.branches:
            identifier = self._candidate_identifier(branch.candidate)
            candidate_statistics = self._sequential_branch_score(branch)
            candidate_values = candidate_statistics["base_values"]
            candidate_mean = candidate_statistics["mean_marginal_entropy"]
            details = {
                "score": candidate_statistics["score"],
                "mean_marginal_entropy": candidate_mean,
                "mean_view_jsd": candidate_statistics["mean_view_jsd"],
                "context_entropy": candidate_statistics["context_entropy"],
                "information_score": candidate_statistics["information_score"],
                "observations": int(candidate_values.size),
                "knowledge_distance": branch.candidate.knowledge_distance,
                "knowledge_distance_ratio": (
                    branch.candidate.knowledge_distance_ratio
                ),
                "cache_health_score": branch.candidate.cache_health_score,
                "mean_advantage": 0.0,
                "standard_error": 0.0,
                "advantage_lcb": 0.0,
                "margin": 0.0,
            }
            if branch.candidate.name != "current":
                paired_count = min(current_values.size, candidate_values.size)
                advantages = (
                    current_values[-paired_count:]
                    - candidate_values[-paired_count:]
                )
                diversity_advantage = (
                    self._meta_conf.adainit_sequential_context_weight
                    * (
                        candidate_statistics["context_entropy"]
                        - current_statistics["context_entropy"]
                    )
                    if self._meta_conf.adainit_sequential_score
                    == "view_infomax"
                    else 0.0
                )
                mean_advantage = float(advantages.mean()) + diversity_advantage
                standard_error = float(
                    np.sqrt(
                        (float(advantages.var()) + self._meta_conf.adainit_eps)
                        / paired_count
                    )
                )
                lcb = (
                    mean_advantage
                    - self._meta_conf.adainit_evidence_confidence_scale
                    * standard_error
                )
                margin = self._sequential_candidate_margin(
                    branch.candidate, pending.trigger_reason
                )
                details.update(
                    {
                        "mean_advantage": mean_advantage,
                        "standard_error": standard_error,
                        "advantage_lcb": lcb,
                        "margin": margin,
                    }
                )
                history_entropy_limit = (
                    self._meta_conf.adainit_periodic_history_max_entropy_increase
                    if pending.trigger_reason == "periodic"
                    else self._meta_conf.adainit_history_max_entropy_increase
                )
                violates_source_gate = bool(
                    branch.candidate.name == "source"
                    and self._meta_conf.adainit_source_min_current_entropy >= 0
                    and current_mean
                    < self._meta_conf.adainit_source_min_current_entropy
                )
                violates_history_gate = bool(
                    branch.candidate.name == "history"
                    and history_entropy_limit >= 0
                    and candidate_mean > current_mean + history_entropy_limit
                )
                violates_generic_gate = bool(
                    self._meta_conf.adainit_max_marginal_entropy_increase >= 0
                    and candidate_mean
                    > current_mean
                    + self._meta_conf.adainit_max_marginal_entropy_increase
                )
                if (
                    lcb > margin
                    and not violates_source_gate
                    and not violates_history_gate
                    and not violates_generic_gate
                ):
                    eligible.append((lcb, branch))
            self._last_sequential_evidence_details[identifier] = details
        if not eligible:
            return None
        return max(eligible, key=lambda item: item[0])[1]

    def _advance_sequential_view_selection(
        self,
        batch: Batch,
        instantaneous_signature: DomainSignature,
        number_of_steps: int,
        timer: Timer,
    ) -> Optional[SequentialEvidenceBranch]:
        """Accumulate, then slide, causal evidence without batching the backend."""

        pending = self._pending_sequential_selection
        if pending is None:
            return None
        pending.proposal_signature = self._update_provisional_signature(
            pending.proposal_signature, instantaneous_signature
        )
        cpu_input = batch._x[0].detach().cpu().clone()
        pending.buffered_inputs.append(cpu_input)
        pending.processed_samples += 1

        if not pending.branches:
            if (
                pending.processed_samples
                < self._meta_conf.adainit_evidence_min_samples
            ):
                return None
            self._initialize_sequential_branches(pending)
            replay_inputs = pending.buffered_inputs
            pending.buffered_inputs = []
            self._process_sequential_inputs(
                pending,
                replay_inputs,
                first_observation_index=0,
                number_of_steps=number_of_steps,
                timer=timer,
            )
        else:
            pending.buffered_inputs = []
            self._process_sequential_inputs(
                pending,
                [cpu_input],
                first_observation_index=pending.processed_samples - 1,
                number_of_steps=number_of_steps,
                timer=timer,
            )

        selected = self._select_sequential_evidence_branch(pending)
        if selected is not None:
            self._pending_sequential_selection = None
            return selected
        if (
            pending.processed_samples
            >= self._meta_conf.adainit_evidence_max_samples
        ):
            self._pending_sequential_selection = None
            return next(
                branch
                for branch in pending.branches
                if branch.candidate.name == "current"
            )
        return None

    def _commit_sequential_evidence_branch(
        self, branch: SequentialEvidenceBranch
    ) -> None:
        """Commit exactly one trajectory; no parameter or prediction mixture."""

        if branch.candidate.name != "current":
            self._load_adaptable_state(branch.candidate.adaptable_state)
            self._reset_after_hard_initialization()
        self._candidate_selection_counts[branch.candidate.name] += 1

    def _apply_backend_recovery(self, preserve_runtime_state: bool = False) -> None:
        """Honor backend recovery (SAR) while keeping source/cache state separate."""

        runtime_state = self._backend_runtime_state
        self._load_adaptable_state(self._source_adaptable_state)
        self._reset_backend_optimizer()
        if preserve_runtime_state:
            # Native SAR's run_multiple_steps writes the triggering EMA back after
            # reset(), so a low recovery EMA persists even though model/optimizer
            # are rebuilt. Preserve that exact backend transition here.
            self._backend_runtime_state = runtime_state

    def _record_knowledge_observation(
        self, prediction_logits: torch.Tensor
    ) -> None:
        """Record label-free formal evidence and sparse parameter snapshots."""

        if (
            self._meta_conf.adainit_knowledge_fingerprint == "disabled"
            and self._meta_conf.adainit_cache_health_mode == "disabled"
        ):
            return
        probability = torch.softmax(
            prediction_logits.detach().float(), dim=1
        )[0].cpu()
        self._knowledge_prediction_window.append(probability)
        sample_count = self._stream_index + 1
        half_window = self._meta_conf.adainit_fingerprint_window // 2
        if sample_count % half_window == 0:
            self._knowledge_state_snapshots.append(
                (sample_count, self._capture_adaptable_state())
            )

    def _restart_knowledge_tracking(self, processed_samples: int) -> None:
        """Start a clean fingerprint window after a hard trajectory switch."""

        self._knowledge_prediction_window.clear()
        self._knowledge_state_snapshots.clear()
        self._knowledge_state_snapshots.append(
            (processed_samples, self._capture_adaptable_state())
        )

    def _knowledge_state_at(
        self, sample_count: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        for count, state in reversed(self._knowledge_state_snapshots):
            if count == sample_count:
                return state
        return None

    def _cache_knowledge_metadata(
        self,
    ) -> Dict[str, Any]:
        """Build cache fingerprint and conservative label-free health metadata."""

        sample_count = self._stream_index + 1
        window = self._meta_conf.adainit_fingerprint_window
        current_state = self._capture_adaptable_state()
        start_state = self._knowledge_state_at(sample_count - window)
        middle_state = self._knowledge_state_at(sample_count - window // 2)
        if self._meta_conf.adainit_knowledge_fingerprint == "source_delta":
            fingerprint_start = self._source_adaptable_state
        else:
            fingerprint_start = start_state
        fingerprint = (
            None
            if self._meta_conf.adainit_knowledge_fingerprint == "disabled"
            or fingerprint_start is None
            else build_knowledge_fingerprint(fingerprint_start, current_state)
        )

        fingerprint_statistics = (
            {}
            if fingerprint is None
            else knowledge_fingerprint_statistics(
                fingerprint, eps=self._meta_conf.adainit_eps
            )
        )
        details: Dict[str, float] = dict(fingerprint_statistics)
        health_score = None
        admitted = True
        if self._meta_conf.adainit_cache_health_mode == "infomax_update":
            if (
                len(self._knowledge_prediction_window) < window
                or start_state is None
                or middle_state is None
            ):
                return {
                    "admitted": False,
                    "reason": "health_window_not_ready",
                    "fingerprint": fingerprint,
                    "health_score": None,
                    "health_details": details,
                }
            probabilities = torch.stack(
                list(self._knowledge_prediction_window)
            ).float().clamp_min(1e-12)
            predictive_entropies = -(
                probabilities * probabilities.log()
            ).sum(dim=1)
            context_marginal = probabilities.mean(dim=0).clamp_min(1e-12)
            context_entropy = float(
                (-(context_marginal * context_marginal.log()).sum()).item()
            )
            mean_predictive_entropy = float(
                predictive_entropies.mean().item()
            )
            information_score = max(
                0.0, context_entropy - mean_predictive_entropy
            )
            class_concentration = float(context_marginal.max().item())
            first_half = build_knowledge_fingerprint(
                start_state, middle_state
            )
            second_half = build_knowledge_fingerprint(
                middle_state, current_state
            )
            update_cosine = knowledge_fingerprint_similarity(
                first_half,
                second_half,
                eps=self._meta_conf.adainit_eps,
            )
            health_score = information_score + (
                self._meta_conf.adainit_cache_health_update_weight
                * max(0.0, update_cosine)
            )
            details.update({
                "context_entropy": context_entropy,
                "mean_predictive_entropy": mean_predictive_entropy,
                "information_score": information_score,
                "class_concentration": class_concentration,
                "update_cosine": update_cosine,
                "health_score": health_score,
            })
            admitted = bool(
                information_score
                >= self._meta_conf.adainit_cache_health_min_information
                and class_concentration
                <= self._meta_conf.adainit_cache_health_max_concentration
                and update_cosine
                >= self._meta_conf.adainit_cache_health_min_update_cosine
                and fingerprint_statistics.get("total_norm", 0.0)
                >= self._meta_conf.adainit_cache_health_min_fingerprint_norm
            )
        if (
            self._meta_conf.adainit_knowledge_fingerprint != "disabled"
            and fingerprint is None
        ):
            admitted = False
        return {
            "admitted": admitted,
            "reason": "accepted" if admitted else "health_gate",
            "fingerprint": fingerprint,
            "health_score": health_score,
            "health_details": details,
        }

    def _insert_cache_anchor(
        self,
        signature: Optional[DomainSignature],
        match_radius: Optional[float] = None,
        adaptable_state: Optional[Dict[str, torch.Tensor]] = None,
        optimizer_state: Optional[Dict[str, Any]] = None,
        backend_runtime_state: Optional[Dict[str, Any]] = None,
        knowledge_fingerprint: Optional[KnowledgeFingerprint] = None,
        health_score: Optional[float] = None,
        health_details: Optional[Dict[str, float]] = None,
    ) -> bool:
        if self._meta_conf.adainit_cache_size == 0:
            return False
        self._initialization_cache.insert(
            (
                self._capture_adaptable_state()
                if adaptable_state is None
                else adaptable_state
            ),
            signature,
            self._stream_index,
            optimizer_state=(
                (
                    self._optimizer.state_dict()
                    if optimizer_state is None
                    else optimizer_state
                )
                if self._meta_conf.adainit_cache_optimizer_state
                else None
            ),
            backend_runtime_state=(
                (
                    self._backend_runtime_state
                    if backend_runtime_state is None
                    else backend_runtime_state
                )
                if self._meta_conf.adainit_cache_optimizer_state
                else None
            ),
            match_radius=match_radius,
            knowledge_fingerprint=knowledge_fingerprint,
            health_score=health_score,
            health_details=health_details,
        )
        return True

    def _run_formal_updates(
        self,
        batch: Batch,
        prediction_before_adaptation: Optional[torch.Tensor],
        nbsteps: int,
        model_selection_method: BaseSelection,
        timer: Timer,
    ) -> None:
        stream_prediction = prediction_before_adaptation
        for step in range(1, nbsteps + 1):
            result = self._backend.adapt_step(
                self._model,
                self._optimizer,
                batch,
                self._backend_runtime_state,
                timer,
                random_seed=self._meta_conf.seed,
            )
            if stream_prediction is None:
                stream_prediction = result.yhat.detach()
            model_selection_method.save_state(
                {
                    "model": copy.deepcopy(self._model.state_dict()),
                    "optimizer": copy.deepcopy(self._optimizer.state_dict()),
                    "step": step,
                    "lr": self._meta_conf.lr,
                    "loss": result.loss,
                    # Always evaluate the prediction made by theta_t before any
                    # detection-driven initialization change or formal update.
                    "yhat": stream_prediction,
                },
                current_batch=batch,
            )
            # Match native SAR ordering: save the adapted candidate first, then
            # reset the active trajectory. Last-iterate selection subsequently
            # restores the saved model while the fresh optimizer and triggering
            # EMA are retained.
            if result.reset_requested:
                self._apply_backend_recovery(preserve_runtime_state=True)

    def adapt_and_eval(
        self,
        episodic: bool,
        metrics: Metrics,
        model_selection_method: BaseSelection,
        current_batch: Batch,
        previous_batches: List[Batch],
        logger: Logger,
        timer: Timer,
    ):
        log = functools.partial(logger.log, display=self._meta_conf.debug)
        if episodic:
            # This path is kept internally coherent, although the constructor
            # rejects it for paper runs.
            self.reset()

        if len(current_batch) != 1:
            raise RuntimeError("AdaInit received a non-singleton online batch.")

        # Retain only a small CPU context. It is used solely at trigger steps to
        # average the unlabeled candidate score across recent observations.
        self._selection_buffer.append(current_batch._x[0].detach().cpu().clone())
        if self._meta_conf.adainit_audit_candidate_accuracy:
            self._selection_audit_labels.append(
                current_batch._y[0].detach().cpu().clone()
            )

        model_selection_method.initialize()
        precomputed_prediction = None
        if self._meta_conf.record_preadapted_perf:
            with timer("evaluate_preadapted_performance"):
                with torch.no_grad(), fork_rng_with_seed(self._meta_conf.seed):
                    precomputed_prediction = self._model(current_batch._x)
                metrics.eval_auxiliary_metric(
                    current_batch._y,
                    precomputed_prediction,
                    metric_name="preadapted_accuracy_top1",
                )

        with timer("adainit.drift_detection"):
            instantaneous_signature = self._extract_signature(current_batch._x)
            retrieval_signature = self._signature_window.append(
                instantaneous_signature
            )
            detector_signature = (
                instantaneous_signature
                if self._meta_conf.adainit_detector_signature_source
                == "instantaneous"
                else retrieval_signature
            )
            previous_reference = (
                None
                if self._detector.signature is None
                else self._detector.signature.clone_cpu()
            )
            if (
                self._meta_conf.adainit_require_full_signature_window
                and not self._signature_window.ready
            ):
                decision = DriftDecision(False, None, None)
            else:
                decision = self._detector.observe(
                    detector_signature, self._stream_index
                )

        if self._pending_sequential_selection is not None:
            if decision.transition_confirmed:
                self._pending_sequential_selection.transition_status = (
                    "main_mds_replaced"
                )
            elif decision.transition_rejected:
                self._pending_sequential_selection.transition_status = (
                    "main_mds_retained"
                )
        if decision.transition_confirmed:
            log(
                "\tAdaInit provisional MDS promoted at "
                f"stream_index={self._stream_index}; the old main MDS was replaced.",
                save=True,
            )
        elif decision.transition_rejected:
            log(
                "\tAdaInit provisional MDS rejected at "
                f"stream_index={self._stream_index}; the old main MDS was retained.",
                save=True,
            )

        # The formal state has not processed x_t yet. Capture it only once at
        # the beginning of a possible adaptive shift; discard it if the evidence
        # run breaks. This avoids a per-sample optimizer copy.
        if (
            self._detector.pending_signatures
            and self._pending_pre_shift_state is None
        ):
            self._pending_pre_shift_state = InitializationCandidate(
                "pre_shift",
                self._capture_adaptable_state(),
                optimizer_state=clone_state_to_cpu(self._optimizer.state_dict()),
                backend_runtime_state=clone_state_to_cpu(
                    self._backend_runtime_state
                ),
            )
        elif not self._detector.pending_signatures and not (
            decision.detected and decision.reason == "adaptive"
        ):
            self._pending_pre_shift_state = None

        prediction_before_adaptation = None
        record_knowledge_observation = True
        selection_is_pending = False
        pending_selection_started = False
        sequential_selector = bool(
            self._meta_conf.adainit_initialization_selector
            == "sequential_view_evidence"
        )
        recovery_proposed = bool(
            (
                decision.proposal
                and self._pending_sequential_selection is None
            )
            if sequential_selector
            else decision.detected
        )
        if recovery_proposed:
            self._number_of_detections += 1
            drift_strength_ratio = (
                None
                if decision.score is None
                or decision.threshold is None
                or decision.threshold <= 0
                else decision.score / decision.threshold
            )
            if (
                decision.detected
                and decision.reason == "adaptive"
                and self._meta_conf.adainit_cache_on_detection
            ):
                # Pair the mature state from immediately before the suspected
                # shift with the old segment signature. For one-confirmation
                # detectors the current pre-update state is the same snapshot.
                pre_shift = self._pending_pre_shift_state
                if pre_shift is None:
                    pre_shift = InitializationCandidate(
                        "pre_shift",
                        self._capture_adaptable_state(),
                        optimizer_state=clone_state_to_cpu(
                            self._optimizer.state_dict()
                        ),
                        backend_runtime_state=clone_state_to_cpu(
                            self._backend_runtime_state
                        ),
                    )
                self._insert_cache_anchor(
                    previous_reference,
                    match_radius=decision.threshold,
                    adaptable_state=pre_shift.adaptable_state,
                    optimizer_state=pre_shift.optimizer_state,
                    backend_runtime_state=pre_shift.backend_runtime_state,
                )
            # Test-then-adapt: preserve theta_t's prediction before temporary
            # branches mutate the active model.
            if precomputed_prediction is None:
                with torch.no_grad(), fork_rng_with_seed(self._meta_conf.seed):
                    prediction_before_adaptation = self._model(current_batch._x)
            else:
                prediction_before_adaptation = precomputed_prediction

            with timer("adainit.initialization_selection"):
                if sequential_selector:
                    pending_selection_started = (
                        self._start_sequential_view_selection(
                            decision.reason or "adaptive",
                            drift_strength_ratio,
                            # Source is a normal hard initialization candidate;
                            # its separate margin and entropy gate prevent weak
                            # source resets.  No boundary or corruption label is
                            # used to make this candidate available.
                            include_source=True,
                        )
                    )
                    selection_is_pending = True
                    self._last_candidate_scores = {}
                    self._last_candidate_score_details = {}
                    self._last_oracle_score_details = {}
                    self._last_oracle_surrogate_details = {}
                    selected = InitializationCandidate(
                        "current",
                        self._capture_adaptable_state(),
                    )
                elif (
                    self._meta_conf.adainit_initialization_selector
                    == "oracle_future_accuracy"
                ):
                    selected = self._select_initialization_oracle(
                        current_batch,
                        retrieval_signature,
                        timer,
                    )
                elif (
                    self._meta_conf.adainit_initialization_selector
                    == "causal_horizon_entropy"
                ):
                    pending_selection_started = self._start_causal_hard_selection(
                        retrieval_signature,
                        decision.reason,
                        drift_strength_ratio,
                        include_source=(
                            (
                                decision.reason == "periodic"
                                and self._meta_conf.adainit_periodic_include_source
                                and drift_strength_ratio is not None
                                and drift_strength_ratio
                                >= self._meta_conf.adainit_source_min_drift_ratio
                            )
                            or (
                                decision.reason != "periodic"
                                and drift_strength_ratio is not None
                                and drift_strength_ratio
                                >= self._meta_conf.adainit_source_min_drift_ratio
                            )
                        ),
                    )
                    selection_is_pending = True
                    self._last_candidate_scores = {}
                    self._last_candidate_score_details = {}
                    self._last_oracle_score_details = {}
                    self._last_oracle_surrogate_details = {}
                    selected = InitializationCandidate(
                        "current",
                        self._capture_adaptable_state(),
                    )
                else:
                    selected = self._select_initialization(
                        current_batch,
                        retrieval_signature,
                        timer,
                        trigger_reason=decision.reason,
                        drift_strength_ratio=drift_strength_ratio,
                        include_source=(
                            (
                                decision.reason == "periodic"
                                and self._meta_conf.adainit_periodic_include_source
                                and drift_strength_ratio is not None
                                and drift_strength_ratio
                                >= self._meta_conf.adainit_source_min_drift_ratio
                            )
                            or (
                                decision.reason != "periodic"
                                and drift_strength_ratio is not None
                                and drift_strength_ratio
                                >= self._meta_conf.adainit_source_min_drift_ratio
                            )
                        ),
                    )
                self._load_adaptable_state(selected.adaptable_state)
                # Counterfactual branches never mutate the formal optimizer. A
                # source/history state must start a fresh trajectory. For current,
                # resetting matches the draft equation, while preserving it makes
                # a guarded false trigger a true continuation (explicit ablation).
                if selection_is_pending:
                    pass
                elif selected.name == "current":
                    if self._meta_conf.adainit_reset_current_optimizer:
                        self._reset_backend_optimizer()
                elif (
                    selected.name == "history"
                    and self._meta_conf.adainit_cache_optimizer_state
                    and selected.optimizer_state is not None
                ):
                    self._reset_backend_optimizer(
                        selected.optimizer_state,
                        selected.backend_runtime_state,
                    )
                else:
                    self._reset_after_hard_initialization()
                if not selection_is_pending and selected.name != "current":
                    # The current sample has not been formally adapted yet, so
                    # the clean local-response window starts after all previous
                    # stream samples and can include x_t below.
                    self._restart_knowledge_tracking(self._stream_index)
                if not selection_is_pending:
                    self._candidate_selection_counts[selected.name] += 1
            if decision.detected and decision.reason == "adaptive":
                self._pending_pre_shift_state = None
            score_text = ", ".join(
                (
                    f"{name}={score:.6g}"
                    + "(after_H="
                    f"{self._last_candidate_score_details[name]['entropy_after']:.6g},"
                    "backend_L="
                    f"{self._last_candidate_score_details[name]['backend_loss']:.6g},"
                    "view_JSD="
                    f"{self._last_candidate_score_details[name]['view_jsd']:.6g}"
                    ",context_diversity="
                    f"{self._last_candidate_score_details[name]['context_diversity']:.6g}"
                    + (
                        ""
                        if self._last_candidate_score_details[name][
                            "audit_view_ensemble_accuracy"
                        ]
                        is None
                        else (
                            ",audit_view_acc="
                            f"{self._last_candidate_score_details[name]['audit_view_ensemble_accuracy']:.6g}"
                        )
                    )
                    + (
                        ")"
                        if self._last_candidate_score_details[name][
                            "entropy_before"
                        ]
                        is None
                        else (
                            ",before_H="
                            f"{self._last_candidate_score_details[name]['entropy_before']:.6g})"
                        )
                    )
                    + (
                        ""
                        if self._last_candidate_score_details[name][
                            "signature_distance"
                        ]
                        is None
                        else (
                            ",signature_distance="
                            f"{self._last_candidate_score_details[name]['signature_distance']:.6g}"
                            + (
                                ""
                                if self._last_candidate_score_details[name][
                                    "cache_insertion_time"
                                ]
                                is None
                                else (
                                    ",cache_time="
                                    f"{self._last_candidate_score_details[name]['cache_insertion_time']}"
                                )
                            )
                            + (
                                ""
                                if self._last_candidate_score_details[name][
                                    "signature_distance_ratio"
                                ]
                                is None
                                else (
                                    ",signature_ratio="
                                    f"{self._last_candidate_score_details[name]['signature_distance_ratio']:.6g}"
                                )
                            )
                            + (
                                ""
                                if self._last_candidate_score_details[name][
                                    "signature_calibrated_ratio"
                                ]
                                is None
                                else (
                                    ",radius_ratio="
                                    f"{self._last_candidate_score_details[name]['signature_calibrated_ratio']:.6g}"
                                )
                            )
                        )
                    )
                )
                for name, score in getattr(self, "_last_candidate_scores", {}).items()
            )
            oracle_text = ", ".join(
                (
                    f"{name}={details['accuracy']:.6g}%"
                    f"/{int(details['samples'])}"
                    f"/mean_H={details['mean_predictive_entropy']:.6g}"
                    f"/marginal_H={details['marginal_predictive_entropy']:.6g}"
                    f"/IM={details['information_maximization_score']:.6g}"
                    f"/stream_L={details['mean_backend_loss']:.6g}"
                    + (
                        ""
                        if "evidence_samples" not in details
                        else (
                            f"/evidence_n={int(details['evidence_samples'])}"
                            f"/evidence_H={details['evidence_mean_marginal_entropy']:.6g}"
                            f"/evidence_JSD={details['evidence_mean_view_jsd']:.6g}"
                            f"/evidence_context_H={details['evidence_context_entropy']:.6g}"
                            f"/evidence_IM={details['evidence_information_maximization_score']:.6g}"
                        )
                    )
                )
                for name, details in self._last_oracle_score_details.items()
            )
            oracle_surrogate_text = ", ".join(
                (
                    f"{name}:score={details['score']:.6g}"
                    f"/post_H={details['entropy_after']:.6g}"
                    f"/backend_L={details['backend_loss']:.6g}"
                    f"/view_JSD={details['view_jsd']:.6g}"
                    + (
                        ""
                        if details["entropy_before"] is None
                        else f"/pre_H={details['entropy_before']:.6g}"
                    )
                )
                for name, details in self._last_oracle_surrogate_details.items()
            )
            selected_text = selected.name
            if selection_is_pending:
                if sequential_selector:
                    pending = self._pending_sequential_selection
                    selected_text = (
                        "pending_sequential_evidence"
                        f"(status={'started' if pending_selection_started else 'already_active'},"
                        "active=current,"
                        f"window={self._meta_conf.adainit_evidence_horizon},"
                        f"max_samples={self._meta_conf.adainit_evidence_max_samples},"
                        f"trigger_time={None if pending is None else pending.trigger_index})"
                    )
                else:
                    pending = self._pending_hard_selection
                    selected_text = (
                        "pending_hard_evidence"
                        f"(status={'started' if pending_selection_started else 'already_active'},"
                        f"active=current,horizon={self._meta_conf.adainit_evidence_horizon},"
                        f"trigger_time={None if pending is None else pending.trigger_index})"
                    )
            if selected.cache_insertion_time is not None:
                selected_text += (
                    f"(cache_time={selected.cache_insertion_time},"
                    f"signature_distance={selected.signature_distance:.6g},"
                    f"signature_ratio={selected.signature_distance_ratio:.6g}"
                )
                if selected.signature_calibrated_ratio is not None:
                    selected_text += (
                        f",radius_ratio={selected.signature_calibrated_ratio:.6g}"
                    )
                selected_text += ")"
            log(
                f"\tAdaInit recovery proposed at stream_index={self._stream_index} "
                f"(reason={decision.reason}, score={decision.score:.6g}, "
                f"threshold={decision.threshold if decision.threshold is not None else 'n/a'}, "
                f"buffer_samples={len(self._selection_buffer)}); "
                f"selected={selected_text}"
                + (f"; candidate_scores=[{score_text}]" if score_text else "")
                + (
                    f"; ORACLE_FUTURE_LABEL_ACCURACY=[{oracle_text}]"
                    if oracle_text
                    else ""
                )
                + (
                    f"; ORACLE_CAUSAL_SURROGATES=[{oracle_surrogate_text}]"
                    if oracle_surrogate_text
                    else ""
                )
                + ".",
                save=True,
            )

        with timer("test_time_adaptation"):
            nbsteps = self._get_adaptation_steps(index=len(previous_batches))
            self._run_formal_updates(
                current_batch,
                prediction_before_adaptation,
                nbsteps,
                model_selection_method,
                timer,
            )

        optimal_state = model_selection_method.select_state()
        self._model.load_state_dict(optimal_state["model"])
        model_selection_method.clean_up()
        if self._oracle_model_selection:
            self.oracle_adaptation_steps.append(optimal_state["step"])
            self._optimizer.load_state_dict(optimal_state["optimizer"])

        metrics.eval(current_batch._y, optimal_state["yhat"])
        with timer("adainit.native_anchor_update"):
            self._advance_native_anchor(current_batch, nbsteps, timer)

        if self._pending_sequential_selection is not None:
            pending = self._pending_sequential_selection
            pending_trigger_index = pending.trigger_index
            pending_trigger_reason = pending.trigger_reason
            transition_status = pending.transition_status
            with timer("adainit.sequential_view_selection"):
                committed_sequential_branch = (
                    self._advance_sequential_view_selection(
                        current_batch,
                        instantaneous_signature,
                        nbsteps,
                        timer,
                    )
                )
                if committed_sequential_branch is not None:
                    self._commit_sequential_evidence_branch(
                        committed_sequential_branch
                    )
                    if committed_sequential_branch.candidate.name != "current":
                        # Branch evidence was computed on a counterfactual
                        # trajectory.  The formal yhat below belongs to the old
                        # current trajectory, so neither it nor pre-switch
                        # snapshots may enter the new cache fingerprint.
                        self._restart_knowledge_tracking(
                            self._stream_index + 1
                        )
                        record_knowledge_observation = False
            if committed_sequential_branch is not None:
                committed_identifier = self._candidate_identifier(
                    committed_sequential_branch.candidate
                )
                evidence_text = ", ".join(
                    (
                        f"{name}=R:{details['score']:.6g}"
                        f"/H:{details['mean_marginal_entropy']:.6g}"
                        f"/JSD:{details['mean_view_jsd']:.6g}"
                        f"/context_H:{details['context_entropy']:.6g}"
                        f"/IM:{details['information_score']:.6g}"
                        f"/adv:{details['mean_advantage']:.6g}"
                        f"/LCB:{details['advantage_lcb']:.6g}"
                        f"/margin:{details['margin']:.6g}"
                        f"/n:{details['observations']}"
                        + (
                            ""
                            if details["knowledge_distance"] is None
                            else (
                                f"/Kdist:{details['knowledge_distance']:.6g}"
                                f"/Kratio:{details['knowledge_distance_ratio']:.6g}"
                                + (
                                    ""
                                    if details["cache_health_score"] is None
                                    else f"/Q:{details['cache_health_score']:.6g}"
                                )
                            )
                        )
                    )
                    for name, details in (
                        self._last_sequential_evidence_details.items()
                    )
                )
                log(
                    "\tAdaInit sequential hard decision committed at "
                    f"stream_index={self._stream_index} "
                    f"(trigger_index={pending_trigger_index},"
                    f"reason={pending_trigger_reason},"
                    f"mds_transition={transition_status}); "
                    f"selected={committed_identifier}; "
                    f"branch_evidence=[{evidence_text}].",
                    save=True,
                )

        if self._pending_hard_selection is not None:
            pending_trigger_index = self._pending_hard_selection.trigger_index
            pending_trigger_reason = self._pending_hard_selection.trigger_reason
            with timer("adainit.causal_hard_selection"):
                committed_branch = self._advance_causal_hard_selection(
                    current_batch,
                    nbsteps,
                    timer,
                )
                if committed_branch is not None:
                    self._commit_causal_evidence_branch(committed_branch)
                    if committed_branch.candidate.name != "current":
                        self._restart_knowledge_tracking(
                            self._stream_index + 1
                        )
                        record_knowledge_observation = False
            if committed_branch is not None:
                committed_identifier = self._candidate_identifier(
                    committed_branch.candidate
                )
                evidence_text = ", ".join(
                    (
                        f"{name}=H:{details['mean_marginal_entropy']:.6g}"
                        f"/JSD:{details['mean_view_jsd']:.6g}"
                        f"/n:{details['observations']}"
                    )
                    for name, details in self._last_causal_evidence_details.items()
                )
                log(
                    "\tAdaInit causal hard decision committed at "
                    f"stream_index={self._stream_index} "
                    f"(trigger_index={pending_trigger_index},"
                    f"reason={pending_trigger_reason},"
                    f"horizon={self._meta_conf.adainit_evidence_horizon}); "
                    f"selected={committed_identifier}; "
                    f"branch_evidence=[{evidence_text}].",
                    save=True,
                )

        if record_knowledge_observation:
            self._record_knowledge_observation(optimal_state["yhat"])

        periodic_health_admission = bool(
            self._meta_conf.adainit_cache_admission == "periodic_health"
        )
        cache_due = bool(
            self._pending_sequential_selection is None
            and (
                (
                    periodic_health_admission
                    and (self._stream_index + 1)
                    % self._meta_conf.adainit_cache_insert_interval
                    == 0
                )
                or (
                    not periodic_health_admission
                    and self._signature_window.ready
                    and not self._detector.pending_signatures
                    and self._detector.segment_length
                    >= self._meta_conf.adainit_cache_min_segment_samples
                    and self._stream_index
                    % self._meta_conf.adainit_cache_insert_interval
                    == 0
                )
            )
        )
        if cache_due and periodic_health_admission:
            cache_metadata = self._cache_knowledge_metadata()
            health_details = cache_metadata["health_details"]
            health_text = ",".join(
                f"{name}={value:.6g}"
                for name, value in health_details.items()
            )
            if cache_metadata["admitted"]:
                inserted = self._insert_cache_anchor(
                    None,
                    knowledge_fingerprint=cache_metadata["fingerprint"],
                    health_score=cache_metadata["health_score"],
                    health_details=health_details,
                )
                if inserted:
                    log(
                        "\tAdaInit knowledge cache admitted at "
                        f"stream_index={self._stream_index}; "
                        f"cache_size={len(self._initialization_cache)}; "
                        f"health=[{health_text}].",
                        save=True,
                    )
            else:
                log(
                    "\tAdaInit knowledge cache rejected at "
                    f"stream_index={self._stream_index}; "
                    f"reason={cache_metadata['reason']}; "
                    f"health=[{health_text}].",
                    save=True,
                )
        elif cache_due:
            cache_signature = (
                retrieval_signature
                if self._meta_conf.adainit_cache_signature_source == "window"
                else self._detector.signature
            )
            instantaneous_radius = (
                None
                if cache_signature is None
                else float(
                    signature_distance(
                        retrieval_signature,
                        cache_signature,
                        self._meta_conf.adainit_eps,
                        self._meta_conf.adainit_signature_distance,
                    ).item()
                )
            )
            statistical_radius = (
                None
                if self._detector.score_mean is None
                else (
                    self._detector.score_mean
                    + self._meta_conf.adainit_drift_beta
                    * self._detector.score_variance ** 0.5
                )
            )
            cache_radius = max(
                value
                for value in (instantaneous_radius, statistical_radius)
                if value is not None
            )
            self._insert_cache_anchor(cache_signature, cache_radius)
        self._stream_index += 1

    def _get_checkpoint_extra_state(self) -> Dict[str, Any]:
        return {
            "detector": self._detector.state_dict(),
            "initialization_cache": self._initialization_cache.state_dict(),
            "signature_window": self._signature_window.state_dict(),
            "selection_buffer": list(self._selection_buffer),
            "selection_audit_labels": list(self._selection_audit_labels),
            "knowledge_prediction_window": list(
                self._knowledge_prediction_window
            ),
            "knowledge_state_snapshots": list(
                self._knowledge_state_snapshots
            ),
            "pending_pre_shift_state": (
                None
                if self._pending_pre_shift_state is None
                else {
                    "adaptable_state": self._pending_pre_shift_state.adaptable_state,
                    "optimizer_state": self._pending_pre_shift_state.optimizer_state,
                    "backend_runtime_state": (
                        self._pending_pre_shift_state.backend_runtime_state
                    ),
                }
            ),
            "pending_hard_selection": (
                None
                if self._pending_hard_selection is None
                else {
                    "trigger_index": self._pending_hard_selection.trigger_index,
                    "trigger_reason": self._pending_hard_selection.trigger_reason,
                    "drift_strength_ratio": (
                        self._pending_hard_selection.drift_strength_ratio
                    ),
                    "processed_samples": (
                        self._pending_hard_selection.processed_samples
                    ),
                    "branches": [
                        {
                            "candidate": {
                                "name": branch.candidate.name,
                                "adaptable_state": (
                                    branch.candidate.adaptable_state
                                ),
                                "signature_distance": (
                                    branch.candidate.signature_distance
                                ),
                                "signature_distance_ratio": (
                                    branch.candidate.signature_distance_ratio
                                ),
                                "cache_insertion_time": (
                                    branch.candidate.cache_insertion_time
                                ),
                                "signature_calibrated_ratio": (
                                    branch.candidate.signature_calibrated_ratio
                                ),
                                "knowledge_distance": (
                                    branch.candidate.knowledge_distance
                                ),
                                "knowledge_distance_ratio": (
                                    branch.candidate.knowledge_distance_ratio
                                ),
                                "cache_health_score": (
                                    branch.candidate.cache_health_score
                                ),
                            },
                            "optimizer_state": branch.optimizer_state,
                            "backend_runtime_state": (
                                branch.backend_runtime_state
                            ),
                            "entropy_sum": branch.entropy_sum,
                            "view_jsd_sum": branch.view_jsd_sum,
                            "observations": branch.observations,
                        }
                        for branch in self._pending_hard_selection.branches
                    ],
                }
            ),
            "pending_sequential_selection": (
                None
                if self._pending_sequential_selection is None
                else {
                    "trigger_index": (
                        self._pending_sequential_selection.trigger_index
                    ),
                    "trigger_reason": (
                        self._pending_sequential_selection.trigger_reason
                    ),
                    "drift_strength_ratio": (
                        self._pending_sequential_selection.drift_strength_ratio
                    ),
                    "trigger_current": {
                        "adaptable_state": (
                            self._pending_sequential_selection.trigger_current.adaptable_state
                        ),
                        "optimizer_state": (
                            self._pending_sequential_selection.trigger_current.optimizer_state
                        ),
                        "backend_runtime_state": (
                            self._pending_sequential_selection.trigger_current.backend_runtime_state
                        ),
                    },
                    "proposal_signature": (
                        None
                        if self._pending_sequential_selection.proposal_signature
                        is None
                        else self._pending_sequential_selection.proposal_signature.state_dict()
                    ),
                    "include_source": (
                        self._pending_sequential_selection.include_source
                    ),
                    "buffered_inputs": (
                        self._pending_sequential_selection.buffered_inputs
                    ),
                    "processed_samples": (
                        self._pending_sequential_selection.processed_samples
                    ),
                    "transition_status": (
                        self._pending_sequential_selection.transition_status
                    ),
                    "branches": [
                        {
                            "candidate": {
                                "name": branch.candidate.name,
                                "adaptable_state": (
                                    branch.candidate.adaptable_state
                                ),
                                "signature_distance": (
                                    branch.candidate.signature_distance
                                ),
                                "signature_distance_ratio": (
                                    branch.candidate.signature_distance_ratio
                                ),
                                "cache_insertion_time": (
                                    branch.candidate.cache_insertion_time
                                ),
                                "signature_calibrated_ratio": (
                                    branch.candidate.signature_calibrated_ratio
                                ),
                                "knowledge_distance": (
                                    branch.candidate.knowledge_distance
                                ),
                                "knowledge_distance_ratio": (
                                    branch.candidate.knowledge_distance_ratio
                                ),
                                "cache_health_score": (
                                    branch.candidate.cache_health_score
                                ),
                            },
                            "optimizer_state": branch.optimizer_state,
                            "backend_runtime_state": (
                                branch.backend_runtime_state
                            ),
                            "marginal_entropies": list(
                                branch.marginal_entropies
                            ),
                            "view_jsds": list(branch.view_jsds),
                            "marginal_probabilities": list(
                                branch.marginal_probabilities
                            ),
                            "observations": branch.observations,
                        }
                        for branch in self._pending_sequential_selection.branches
                    ],
                }
            ),
            "backend_runtime_state": self._backend_runtime_state,
            "stream_index": self._stream_index,
            "number_of_detections": self._number_of_detections,
            "candidate_selection_counts": self._candidate_selection_counts,
            "native_anchor_state": self._native_anchor_state,
            "native_anchor_optimizer_state": (
                None
                if self._native_anchor_optimizer is None
                else self._native_anchor_optimizer.state_dict()
            ),
            "native_anchor_runtime_state": self._native_anchor_runtime_state,
        }

    def _load_checkpoint_extra_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self._detector.load_state_dict(state["detector"])
        self._initialization_cache.load_state_dict(state["initialization_cache"])
        self._signature_window.load_state_dict(state.get("signature_window", {}))
        self._selection_buffer = deque(
            state.get("selection_buffer", []),
            maxlen=self._meta_conf.adainit_selection_window,
        )
        self._selection_audit_labels = deque(
            state.get("selection_audit_labels", []),
            maxlen=self._meta_conf.adainit_selection_window,
        )
        self._knowledge_prediction_window = deque(
            state.get("knowledge_prediction_window", []),
            maxlen=self._meta_conf.adainit_fingerprint_window,
        )
        self._knowledge_state_snapshots = deque(
            state.get(
                "knowledge_state_snapshots",
                [(0, clone_state_to_cpu(self._source_adaptable_state))],
            ),
            maxlen=4,
        )
        pending_pre_shift_state = state.get("pending_pre_shift_state")
        self._pending_pre_shift_state = (
            None
            if pending_pre_shift_state is None
            else InitializationCandidate(
                "pre_shift",
                pending_pre_shift_state["adaptable_state"],
                optimizer_state=pending_pre_shift_state.get("optimizer_state"),
                backend_runtime_state=pending_pre_shift_state.get(
                    "backend_runtime_state"
                ),
            )
        )
        pending_hard_selection = state.get("pending_hard_selection")
        self._pending_hard_selection = (
            None
            if pending_hard_selection is None
            else PendingHardSelection(
                trigger_index=pending_hard_selection["trigger_index"],
                trigger_reason=pending_hard_selection["trigger_reason"],
                drift_strength_ratio=pending_hard_selection.get(
                    "drift_strength_ratio"
                ),
                processed_samples=pending_hard_selection.get(
                    "processed_samples", 0
                ),
                branches=[
                    CausalEvidenceBranch(
                        candidate=InitializationCandidate(
                            name=item["candidate"]["name"],
                            adaptable_state=item["candidate"][
                                "adaptable_state"
                            ],
                            signature_distance=item["candidate"].get(
                                "signature_distance"
                            ),
                            signature_distance_ratio=item["candidate"].get(
                                "signature_distance_ratio"
                            ),
                            cache_insertion_time=item["candidate"].get(
                                "cache_insertion_time"
                            ),
                            signature_calibrated_ratio=item["candidate"].get(
                                "signature_calibrated_ratio"
                            ),
                            knowledge_distance=item["candidate"].get(
                                "knowledge_distance"
                            ),
                            knowledge_distance_ratio=item["candidate"].get(
                                "knowledge_distance_ratio"
                            ),
                            cache_health_score=item["candidate"].get(
                                "cache_health_score"
                            ),
                        ),
                        optimizer_state=item["optimizer_state"],
                        backend_runtime_state=item["backend_runtime_state"],
                        entropy_sum=item.get("entropy_sum", 0.0),
                        view_jsd_sum=item.get("view_jsd_sum", 0.0),
                        observations=item.get("observations", 0),
                    )
                    for item in pending_hard_selection.get("branches", [])
                ],
            )
        )
        pending_sequential = state.get("pending_sequential_selection")
        if pending_sequential is None:
            self._pending_sequential_selection = None
        else:
            trigger_current = pending_sequential["trigger_current"]
            proposal_signature = pending_sequential.get("proposal_signature")
            self._pending_sequential_selection = PendingSequentialSelection(
                trigger_index=pending_sequential["trigger_index"],
                trigger_reason=pending_sequential["trigger_reason"],
                drift_strength_ratio=pending_sequential.get(
                    "drift_strength_ratio"
                ),
                trigger_current=InitializationCandidate(
                    "current",
                    trigger_current["adaptable_state"],
                    optimizer_state=trigger_current.get("optimizer_state"),
                    backend_runtime_state=trigger_current.get(
                        "backend_runtime_state"
                    ),
                ),
                proposal_signature=(
                    None
                    if proposal_signature is None
                    else DomainSignature.from_state_dict(proposal_signature)
                ),
                include_source=pending_sequential.get(
                    "include_source", True
                ),
                buffered_inputs=list(
                    pending_sequential.get("buffered_inputs", [])
                ),
                processed_samples=pending_sequential.get(
                    "processed_samples", 0
                ),
                transition_status=pending_sequential.get(
                    "transition_status", "pending"
                ),
                branches=[
                    SequentialEvidenceBranch(
                        candidate=InitializationCandidate(
                            name=item["candidate"]["name"],
                            adaptable_state=item["candidate"][
                                "adaptable_state"
                            ],
                            signature_distance=item["candidate"].get(
                                "signature_distance"
                            ),
                            signature_distance_ratio=item["candidate"].get(
                                "signature_distance_ratio"
                            ),
                            cache_insertion_time=item["candidate"].get(
                                "cache_insertion_time"
                            ),
                            signature_calibrated_ratio=item["candidate"].get(
                                "signature_calibrated_ratio"
                            ),
                            knowledge_distance=item["candidate"].get(
                                "knowledge_distance"
                            ),
                            knowledge_distance_ratio=item["candidate"].get(
                                "knowledge_distance_ratio"
                            ),
                            cache_health_score=item["candidate"].get(
                                "cache_health_score"
                            ),
                        ),
                        optimizer_state=item["optimizer_state"],
                        backend_runtime_state=item["backend_runtime_state"],
                        marginal_entropies=deque(
                            item.get("marginal_entropies", []),
                            maxlen=self._meta_conf.adainit_evidence_horizon,
                        ),
                        view_jsds=deque(
                            item.get("view_jsds", []),
                            maxlen=self._meta_conf.adainit_evidence_horizon,
                        ),
                        marginal_probabilities=deque(
                            item.get("marginal_probabilities", []),
                            maxlen=self._meta_conf.adainit_evidence_horizon,
                        ),
                        observations=item.get("observations", 0),
                    )
                    for item in pending_sequential.get("branches", [])
                ],
            )
        self._backend_runtime_state = state.get(
            "backend_runtime_state", self._backend.fresh_runtime_state()
        )
        self._stream_index = state.get("stream_index", 0)
        self._number_of_detections = state.get("number_of_detections", 0)
        self._candidate_selection_counts = state.get(
            "candidate_selection_counts", self._candidate_selection_counts
        )
        if self._meta_conf.adainit_native_anchor:
            self._native_anchor_state = state.get(
                "native_anchor_state", self._native_anchor_state
            )
            native_anchor_optimizer_state = state.get(
                "native_anchor_optimizer_state"
            )
            if native_anchor_optimizer_state is not None:
                self._native_anchor_optimizer.load_state_dict(
                    native_anchor_optimizer_state
                )
            self._native_anchor_runtime_state = state.get(
                "native_anchor_runtime_state",
                self._native_anchor_runtime_state,
            )

    @property
    def name(self):
        return "adainit"
