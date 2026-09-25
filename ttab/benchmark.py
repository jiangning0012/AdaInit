# -*- coding: utf-8 -*-
import copy
import json
import os
import random
import resource
import time
from collections import defaultdict, deque
from itertools import islice
from typing import Any, Dict, List, Type, Union

import numpy as np
import torch
import ttab.scenarios as scenarios
import ttab.utils.auxiliary as auxiliary
import ttab.utils.checkpoint as checkpoint
from ttab.api import Batch, PyTorchDataset
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer

D = Union[torch.utils.data.Dataset, PyTorchDataset]


class Benchmark(object):
    def __init__(
        self,
        scenario: scenarios.Scenario,
        model_adaptation_cls: Type[BaseAdaptation],
        model_selection_cls: Type[BaseSelection],
        test_loader: D,
        meta_conf: Dict,
    ) -> None:
        # assign variables.
        self._scenario = scenario
        self._model_adaptation_cls = model_adaptation_cls
        self._model_selection_cls = model_selection_cls
        self._test_loader = test_loader
        self._meta_conf = copy.deepcopy(meta_conf)

        # init.
        self._safety_check()
        self._init_benchmark()

    def _safety_check(self) -> None:
        assert hasattr(self._meta_conf, "seed")
        assert hasattr(self._meta_conf, "root_path")

        # assign variables for convenience.
        self._meta_conf.device = (
            "cuda" if not hasattr(self._meta_conf, "device") else self._meta_conf.device
        )
        if not hasattr(self._meta_conf, "record_first_n_per_domain"):
            self._meta_conf.record_first_n_per_domain = 1000
        if self._meta_conf.record_first_n_per_domain < 0:
            raise ValueError("record_first_n_per_domain must be greater than or equal to 0")
        if not hasattr(self._meta_conf, "checkpoint_every_domains"):
            self._meta_conf.checkpoint_every_domains = 5
        if self._meta_conf.checkpoint_every_domains < 0:
            raise ValueError("checkpoint_every_domains must be greater than or equal to 0")
        if not hasattr(self._meta_conf, "domain_replay_visits"):
            self._meta_conf.domain_replay_visits = 1
        if not hasattr(self._meta_conf, "domain_replay_samples_per_visit"):
            self._meta_conf.domain_replay_samples_per_visit = 0
        if not hasattr(self._meta_conf, "reset_adaptation_on_domain_boundary"):
            self._meta_conf.reset_adaptation_on_domain_boundary = False
        if self._meta_conf.domain_replay_visits < 1:
            raise ValueError("domain_replay_visits must be at least 1")
        if self._meta_conf.domain_replay_samples_per_visit < 0:
            raise ValueError(
                "domain_replay_samples_per_visit must be greater than or equal to 0"
            )

    def _init_benchmark(self) -> None:
        # init logging.
        self._checkpoint_path: str = checkpoint.init_checkpoint(self._meta_conf)
        self._logger = Logger(folder_path=self._checkpoint_path)
        # init metrics.
        self._metrics = Metrics(self._scenario)
        if self._meta_conf.record_preadapted_perf:
            self._metrics.init_auxiliary_metric(metric_name="preadapted_accuracy_top1")

        # init timer.
        self._timer = Timer(
            device=self._meta_conf.device,
            verbosity_level=1
            if not hasattr(self._meta_conf, "track_time")
            else self._meta_conf.track_time,
            log_fn=self._logger.log_metric,
            on_cuda=True if "cuda" in self._meta_conf.device else False,
        )

        # Domain boundaries are inferred from the actual sampled child datasets.
        # This remains correct when domain_sampling_ratio or data_size changes.
        self._domain_schedule = self._build_domain_schedule()
        self._domain_summaries = []
        self._active_domain = None
        self._resume_domain_index = 0
        self._resume_processed_samples = 0
        self._resume_last_step = 0
        self._resume_previous_batches_count = 0
        self._resume_stream_seconds = 0.0
        self._resumed_from = None
        if getattr(self._meta_conf, "resume_checkpoint", None):
            self._load_recovery_checkpoint(self._meta_conf.resume_checkpoint)
        self._initial_storage = self._storage_snapshot()

    def _checkpoint_signature(self) -> Dict[str, Any]:
        signature = {
            "model_name": self._scenario.model_name,
            "model_adaptation_method": self._scenario.model_adaptation_method,
            "model_selection_method": self._scenario.model_selection_method,
            "data_names": self._meta_conf.data_names,
            "data_wise": self._scenario.test_case.data_wise,
            "batch_size": self._scenario.test_case.batch_size,
            "episodic": self._scenario.test_case.episodic,
            "intra_domain_shuffle": self._scenario.test_case.intra_domain_shuffle,
            "inter_domain": type(
                self._scenario.test_case.inter_domain
            ).__name__,
            "seed": self._meta_conf.seed,
            "lr": self._meta_conf.lr,
            "n_train_steps": self._meta_conf.n_train_steps,
            "fishers": self._meta_conf.fishers,
            "record_first_n_per_domain": (
                self._meta_conf.record_first_n_per_domain
            ),
            "update_on_domain_partial_batch": getattr(
                self._meta_conf, "update_on_domain_partial_batch", False
            ),
            "domain_replay_visits": self._meta_conf.domain_replay_visits,
            "domain_replay_samples_per_visit": (
                self._meta_conf.domain_replay_samples_per_visit
            ),
            "reset_adaptation_on_domain_boundary": (
                self._meta_conf.reset_adaptation_on_domain_boundary
            ),
        }
        if self._scenario.model_adaptation_method == "adainit":
            adainit_keys = [
                "adainit_backend",
                "adainit_signature_momentum",
                "adainit_signature_window_size",
                "adainit_signature_distance",
                "adainit_detector_distance",
                "adainit_detector_signature_source",
                "adainit_score_momentum",
                "adainit_drift_beta",
                "adainit_fixed_drift_threshold",
                "adainit_drift_min_ratio",
                "adainit_eps",
                "adainit_min_reference_samples",
                "adainit_drift_confirmations",
                "adainit_detector_mode",
                "adainit_periodic_interval",
                "adainit_periodic_include_source",
                "adainit_trigger_cooldown",
                "adainit_cache_size",
                "adainit_cache_eviction",
                "adainit_cache_admission",
                "adainit_knowledge_fingerprint",
                "adainit_fingerprint_window",
                "adainit_fingerprint_magnitude_weight",
                "adainit_cache_insert_interval",
                "adainit_cache_signature_source",
                "adainit_cache_min_segment_samples",
                "adainit_cache_on_detection",
                "adainit_cache_optimizer_state",
                "adainit_cache_health_mode",
                "adainit_cache_health_min_information",
                "adainit_cache_health_max_concentration",
                "adainit_cache_health_min_update_cosine",
                "adainit_cache_health_min_fingerprint_norm",
                "adainit_cache_health_update_weight",
                "adainit_history_retrieval",
                "adainit_history_fingerprint_max_distance",
                "adainit_history_fingerprint_max_ratio",
                "adainit_max_history_candidates",
                "adainit_history_min_age",
                "adainit_history_match_ratio",
                "adainit_history_max_distance",
                "adainit_history_radius_multiplier",
                "adainit_source_for_unseen_only",
                "adainit_source_min_drift_ratio",
                "adainit_source_min_current_entropy",
                "adainit_history_score_bonus",
                "adainit_history_min_drift_ratio",
                "adainit_context_diversity_weight",
                "adainit_history_max_entropy_increase",
                "adainit_periodic_selection_margin",
                "adainit_periodic_history_max_entropy_increase",
                "adainit_prefer_matched_history",
                "adainit_num_views",
                "adainit_selection_window",
                "adainit_counterfactual_steps",
                "adainit_selection_margin",
                "adainit_source_selection_margin",
                "adainit_reset_current_optimizer",
                "adainit_max_marginal_entropy_increase",
                "adainit_recovery_score",
                "adainit_candidate_mode",
                "adainit_initialization_selector",
                "adainit_evidence_horizon",
                "adainit_evidence_min_samples",
                "adainit_evidence_max_samples",
                "adainit_evidence_confidence_scale",
                "adainit_evidence_warmup",
                "adainit_evidence_timing",
                "adainit_sequential_score",
                "adainit_sequential_view_jsd_weight",
                "adainit_sequential_context_weight",
                "adainit_oracle_horizon",
                "adainit_oracle_max_history_candidates",
                "adainit_oracle_include_source",
                "adainit_oracle_min_accuracy_gain",
                "adainit_oracle_log_surrogates",
                "adainit_oracle_log_future_view_surrogates",
                "adainit_native_anchor",
                "adainit_native_anchor_selection_margin",
            ]
            signature["adainit"] = {
                key: getattr(self._meta_conf, key) for key in adainit_keys
            }
        elif self._scenario.model_adaptation_method == "dpcore":
            dpcore_keys = [
                "dpcore_prompt_num",
                "dpcore_threshold_ratio",
                "dpcore_update_weight",
                "dpcore_temperature",
                "dpcore_new_prompt_steps",
                "dpcore_reuse_steps",
                "dpcore_std_weight",
                "dpcore_moment_eps",
                "dpcore_source_num_samples",
                "dpcore_source_stats_path",
            ]
            signature["dpcore"] = {
                key: getattr(self._meta_conf, key) for key in dpcore_keys
            }
        return signature

    def _load_recovery_checkpoint(self, path: str) -> None:
        load_start = time.perf_counter()
        resolved_path, state = checkpoint.load_recovery_checkpoint(path)
        if state.get("format_version") != 1:
            raise RuntimeError(
                f"Unsupported recovery checkpoint version: {state.get('format_version')}."
            )
        expected_signature = self._checkpoint_signature()
        checkpoint_signature = state.get("signature")
        signature_matches = checkpoint_signature == expected_signature
        if (
            not signature_matches
            and isinstance(checkpoint_signature, dict)
            and isinstance(checkpoint_signature.get("adainit"), dict)
            and isinstance(expected_signature.get("adainit"), dict)
        ):
            # Checkpoints written before the knowledge-fingerprint branch do not
            # contain its selector/cache controls.  Accept only that explicitly
            # known omission; all fields present in the old signature must still
            # match exactly.
            backward_compatible_adainit_keys = {
                "adainit_cache_admission",
                "adainit_knowledge_fingerprint",
                "adainit_fingerprint_window",
                "adainit_fingerprint_magnitude_weight",
                "adainit_cache_health_mode",
                "adainit_cache_health_min_information",
                "adainit_cache_health_max_concentration",
                "adainit_cache_health_min_update_cosine",
                "adainit_cache_health_min_fingerprint_norm",
                "adainit_cache_health_update_weight",
                "adainit_history_retrieval",
                "adainit_history_fingerprint_max_distance",
                "adainit_history_fingerprint_max_ratio",
                "adainit_sequential_score",
                "adainit_sequential_view_jsd_weight",
                "adainit_sequential_context_weight",
                "adainit_evidence_min_samples",
                "adainit_evidence_max_samples",
                "adainit_evidence_confidence_scale",
            }
            compatible_expected = copy.deepcopy(expected_signature)
            old_adainit = checkpoint_signature["adainit"]
            missing_keys = set(compatible_expected["adainit"]) - set(old_adainit)
            if missing_keys <= backward_compatible_adainit_keys:
                for key in missing_keys:
                    compatible_expected["adainit"].pop(key)
                signature_matches = checkpoint_signature == compatible_expected
        if not signature_matches:
            raise RuntimeError(
                "Recovery checkpoint configuration mismatch.\n"
                f"checkpoint={checkpoint_signature}\ncurrent={expected_signature}"
            )

        next_domain_index = int(state["next_domain_index"])
        if not 0 <= next_domain_index <= len(self._domain_schedule):
            raise RuntimeError(
                f"Invalid next_domain_index={next_domain_index} in {resolved_path}."
            )

        archived_log_tails = self._logger.truncate_to_offsets(
            state["log_offsets"]
        )
        self._model_adaptation_cls.load_checkpoint_state(state["adaptation_state"])
        self._metrics.tracker.load_state_dict(state["metrics_state"])
        self._domain_summaries = state["domain_summaries"]

        self._timer.totals = dict(state.get("timer_totals", {}))
        self._timer.call_counts = dict(state.get("timer_call_counts", {}))
        timer_now = time.time()
        self._timer.first_time = {
            label: timer_now for label in self._timer.totals
        }
        self._timer.last_time = {label: timer_now for label in self._timer.totals}

        random.setstate(state["python_random_state"])
        np.random.set_state(state["numpy_random_state"])
        torch.set_rng_state(state["torch_random_state"])
        if self._timer.cuda_available and state.get("cuda_random_state") is not None:
            torch.cuda.set_rng_state_all(state["cuda_random_state"])

        self._resume_domain_index = next_domain_index
        self._resume_processed_samples = int(state["processed_samples"])
        self._resume_last_step = int(state["last_step"])
        self._resume_previous_batches_count = int(
            state["previous_batches_count"]
        )
        self._resume_stream_seconds = float(state["stream_seconds_completed"])
        self._resumed_from = resolved_path
        self._meta_conf.checkpoint_load_seconds = time.perf_counter() - load_start
        self._logger.log_metric(
            name="checkpoint",
            values={
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "checkpoint_path": resolved_path,
                "next_domain_index": next_domain_index,
                "processed_samples": self._resume_processed_samples,
                "archived_uncommitted_log_tails": archived_log_tails,
            },
            tags={"split": "test", "type": "resume"},
            display=True,
        )

    def _save_recovery_checkpoint(
        self,
        completed_domains: int,
        processed_samples: int,
        last_step: int,
        previous_batches_count: int,
        stream_seconds_completed: float,
    ) -> str:
        self._sync_cuda()
        state = {
            "format_version": 1,
            "signature": self._checkpoint_signature(),
            "checkpoint_path": self._checkpoint_path,
            "next_domain_index": completed_domains,
            "processed_samples": processed_samples,
            "last_step": last_step,
            "previous_batches_count": previous_batches_count,
            "stream_seconds_completed": stream_seconds_completed,
            "domain_summaries": self._domain_summaries,
            "metrics_state": self._metrics.tracker.state_dict(),
            "adaptation_state": self._model_adaptation_cls.get_checkpoint_state(),
            "timer_totals": dict(self._timer.totals),
            "timer_call_counts": dict(self._timer.call_counts),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all()
            if self._timer.cuda_available
            else None,
            "log_offsets": self._logger.get_offsets(),
        }
        checkpoint_path = checkpoint.save_recovery_checkpoint(
            state=state,
            folder_path=self._checkpoint_path,
            completed_domains=completed_domains,
        )
        self._logger.log_metric(
            name="checkpoint",
            values={
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "checkpoint_path": checkpoint_path,
                "completed_domains": completed_domains,
                "next_domain_index": completed_domains,
            },
            tags={"split": "test", "type": "checkpoint_saved"},
            display=True,
        )
        return checkpoint_path

    def _build_domain_schedule(self) -> List[Dict[str, Any]]:
        dataset = self._test_loader.dataset
        test_domains = list(self._scenario.test_domains)
        child_datasets = getattr(dataset, "datasets", None)

        replay_children = (
            child_datasets is not None
            and len(child_datasets) > 0
            and all(hasattr(child, "replay_metadata") for child in child_datasets)
        )

        if replay_children:
            schedule_entries = []
            self._domain_stream_datasets = list(child_datasets)
            boundary_source = "round_major_replay_child_datasets"
            for child_dataset in child_datasets:
                replay_metadata = dict(child_dataset.replay_metadata)
                base_domain_index = int(replay_metadata["base_domain_index"])
                if not 0 <= base_domain_index < len(test_domains):
                    raise RuntimeError(
                        "Invalid base domain index in replay metadata: "
                        f"{base_domain_index}."
                    )
                schedule_entries.append(
                    (
                        test_domains[base_domain_index],
                        len(child_dataset),
                        replay_metadata,
                    )
                )
        elif (
            child_datasets is not None
            and len(child_datasets) == len(test_domains)
        ):
            schedule_entries = [
                (domain, len(child_dataset), None)
                for domain, child_dataset in zip(test_domains, child_datasets)
            ]
            self._domain_stream_datasets = list(child_datasets)
            boundary_source = "actual_child_dataset_lengths"
        else:
            raise RuntimeError(
                "The ImageNet-C stream lost its explicit contiguous-domain children."
            )

        schedule = []
        start_sample = 1
        for domain_index, (domain, domain_length, replay_metadata) in enumerate(
            schedule_entries
        ):
            shift_property = getattr(domain, "shift_property", None)
            base_domain_name = (
                domain.data_name if domain is not None else "mixed_test_stream"
            )
            replay_visit = (
                int(replay_metadata["replay_visit"])
                if replay_metadata is not None
                else None
            )
            domain_name = (
                f"{base_domain_name}::visit_{replay_visit:02d}"
                if replay_visit is not None
                else base_domain_name
            )
            reset_at_start = bool(
                self._meta_conf.reset_adaptation_on_domain_boundary
            )
            schedule.append(
                {
                    "domain_index": domain_index,
                    "domain_name": domain_name,
                    "base_domain_name": base_domain_name,
                    "corruption": getattr(shift_property, "shift_name", None),
                    "severity": getattr(shift_property, "shift_degree", None),
                    "num_samples": int(domain_length),
                    "start_sample": start_sample,
                    "end_sample": start_sample + int(domain_length) - 1,
                    "is_shift_boundary": domain_index > 0,
                    "boundary_source": boundary_source,
                    "reset_at_start": reset_at_start,
                    **(replay_metadata or {}),
                }
            )
            start_sample += int(domain_length)

        if schedule and schedule[-1]["end_sample"] != len(dataset):
            raise RuntimeError(
                "Domain schedule does not match the constructed test dataset: "
                f"scheduled={schedule[-1]['end_sample']}, actual={len(dataset)}."
            )
        return schedule

    def _sync_cuda(self) -> None:
        if self._timer.cuda_available:
            torch.cuda.synchronize(device=self._meta_conf.device)

    @staticmethod
    def _process_current_rss_bytes() -> int:
        try:
            with open("/proc/self/status", "r") as fp:
                for line in fp:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
        except (OSError, IndexError, ValueError):
            pass
        return 0

    @staticmethod
    def _process_peak_rss_bytes() -> int:
        try:
            with open("/proc/self/status", "r") as fp:
                for line in fp:
                    if line.startswith("VmHWM:"):
                        return int(line.split()[1]) * 1024
        except (OSError, IndexError, ValueError):
            pass
        # Linux reports ru_maxrss in KiB; this is the fallback for /proc-less hosts.
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)

    def _persistent_storage_bytes(self) -> Dict[str, Any]:
        """Count unique tensor/ndarray storage retained by the TTA and selector."""
        seen_objects = set()
        seen_storages = set()
        bytes_by_device = defaultdict(int)

        def visit(value):
            if value is None or isinstance(value, (str, bytes, int, float, bool)):
                return

            object_id = id(value)
            if object_id in seen_objects:
                return
            seen_objects.add(object_id)

            if torch.is_tensor(value):
                try:
                    storage = value.untyped_storage()
                    storage_key = (
                        str(value.device),
                        storage.data_ptr(),
                        storage.nbytes(),
                    )
                    storage_bytes = storage.nbytes()
                except (AttributeError, RuntimeError):
                    storage_key = (str(value.device), object_id, value.numel())
                    storage_bytes = value.numel() * value.element_size()
                if storage_key not in seen_storages:
                    seen_storages.add(storage_key)
                    bytes_by_device[str(value.device)] += int(storage_bytes)
                return

            if isinstance(value, np.ndarray):
                storage_key = ("numpy", object_id, value.nbytes)
                if storage_key not in seen_storages:
                    seen_storages.add(storage_key)
                    bytes_by_device["cpu_numpy"] += int(value.nbytes)
                return

            if isinstance(value, dict):
                for key, item in value.items():
                    visit(key)
                    visit(item)
                return

            if isinstance(value, (list, tuple, set)):
                for item in value:
                    visit(item)
                return

            if isinstance(value, torch.nn.Module):
                for parameter in value.parameters(recurse=True):
                    visit(parameter)
                for buffer in value.buffers(recurse=True):
                    visit(buffer)
                return

            if isinstance(value, torch.optim.Optimizer):
                visit(value.state)

        # Only inspect top-level state owned by the adaptation and selection
        # objects. Avoid recursively walking arbitrary Python internals of timm
        # modules, which is unnecessary for tensor-storage accounting.
        for owner in (self._model_adaptation_cls, self._model_selection_cls):
            for value in vars(owner).values():
                visit(value)
        return {
            "persistent_tensor_storage_bytes": int(sum(bytes_by_device.values())),
            "persistent_tensor_storage_bytes_by_device": dict(bytes_by_device),
        }

    def _storage_snapshot(self, include_persistent: bool = True) -> Dict[str, Any]:
        snapshot = {
            "process_current_rss_bytes": self._process_current_rss_bytes(),
            "process_peak_rss_bytes": self._process_peak_rss_bytes(),
        }
        if include_persistent:
            snapshot.update(self._persistent_storage_bytes())
        if self._timer.cuda_available:
            snapshot.update(
                {
                    "cuda_memory_allocated_bytes": int(
                        torch.cuda.memory_allocated(self._meta_conf.device)
                    ),
                    "cuda_memory_reserved_bytes": int(
                        torch.cuda.memory_reserved(self._meta_conf.device)
                    ),
                    "cuda_peak_memory_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(self._meta_conf.device)
                    ),
                    "cuda_peak_memory_reserved_bytes": int(
                        torch.cuda.max_memory_reserved(self._meta_conf.device)
                    ),
                }
            )

        if include_persistent:
            storage_hook = getattr(
                self._model_adaptation_cls, "get_storage_metrics", None
            )
            if callable(storage_hook):
                extra_storage = storage_hook()
                if isinstance(extra_storage, dict):
                    snapshot["method_reported_storage"] = extra_storage
        return snapshot

    def _start_domain(self, domain_index: int) -> None:
        domain = self._domain_schedule[domain_index]
        self._sync_cuda()
        if self._timer.cuda_available:
            torch.cuda.reset_peak_memory_stats(device=self._meta_conf.device)

        reset_applied = False
        reset_reason = None
        if domain["reset_at_start"]:
            if self._scenario.model_adaptation_method == "no_adaptation":
                reset_reason = "not_required_for_no_adaptation"
            else:
                self._model_adaptation_cls.reset()
                self._model_selection_cls.initialize()
                reset_applied = True
                reset_reason = "known_domain_boundary"
            self._logger.log_metric(
                name="domain",
                values={
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    **domain,
                    "reset_applied": reset_applied,
                    "reset_reason": reset_reason,
                },
                tags={"split": "test", "type": "domain_reset"},
                display=True,
            )

        self._active_domain = {
            "descriptor": domain,
            "start_time": time.perf_counter(),
            "samples": 0,
            "batches": 0,
            "update_batches": 0,
            "prediction_only_batches": 0,
            "prediction_only_samples": 0,
            "metric_sums": defaultdict(float),
            "prefix_metrics": {},
            "batch_prefix_metrics": {},
            "reset_applied": reset_applied,
            "reset_reason": reset_reason,
        }
        transition_from = (
            self._domain_schedule[domain_index - 1]["domain_name"]
            if domain_index > 0
            else None
        )
        self._logger.log_metric(
            name="domain",
            values={
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                **domain,
                "transition_from": transition_from,
            },
            tags={"split": "test", "type": "domain_start"},
            display=True,
        )

    def _update_domain_metrics(
        self,
        step: int,
        epoch: float,
        batch_size: int,
        update_applied: bool,
    ) -> None:
        current_metrics = self._metrics.tracker.get_current_val()
        active = self._active_domain
        previous_samples = active["samples"]
        active["samples"] += batch_size
        active["batches"] += 1
        if update_applied:
            active["update_batches"] += 1
        else:
            active["prediction_only_batches"] += 1
            active["prediction_only_samples"] += batch_size

        for metric_name, metric_value in current_metrics.items():
            active["metric_sums"][metric_name] += metric_value * batch_size

        for prefix_size in (10, 50, 100, 1000):
            if previous_samples < prefix_size <= active["samples"]:
                active["prefix_metrics"][prefix_size] = {
                    metric_name: (
                        active["metric_sums"][metric_name]
                        - metric_value * (active["samples"] - prefix_size)
                    )
                    / prefix_size
                    for metric_name, metric_value in current_metrics.items()
                }

        if (
            self._scenario.test_case.data_wise == "batch_wise"
            and active["batches"] in (10, 50, 100, 1000)
        ):
            active["batch_prefix_metrics"][active["batches"]] = {
                metric_name: metric_sum / active["samples"]
                for metric_name, metric_sum in active["metric_sums"].items()
            }

        record_limit = self._meta_conf.record_first_n_per_domain
        domain_sample = previous_samples + 1
        # In batch-wise mode one record represents the whole batch. Keep batches
        # whose first sample is inside the requested sample prefix; for W32 and
        # N=1000 this produces 32 records (the last one covers samples 993--1024).
        should_record = record_limit > 0 and domain_sample <= record_limit
        if should_record:
            running_metrics = {
                f"domain_running_{metric_name}": metric_sum / active["samples"]
                for metric_name, metric_sum in active["metric_sums"].items()
            }
            descriptor = active["descriptor"]
            self._logger.log_metric(
                name="evaluation",
                values={
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "step": step,
                    "epoch": epoch,
                    "global_sample": descriptor["start_sample"] + previous_samples,
                    "global_sample_end": (
                        descriptor["start_sample"]
                        + previous_samples
                        + batch_size
                        - 1
                    ),
                    "domain_index": descriptor["domain_index"],
                    "domain_name": descriptor["domain_name"],
                    "corruption": descriptor["corruption"],
                    "severity": descriptor["severity"],
                    "domain_sample": domain_sample,
                    "domain_sample_end": previous_samples + batch_size,
                    "domain_batch": active["batches"],
                    "actual_batch_size": batch_size,
                    "update_applied": update_applied,
                    "prediction_phase": "before_adapt",
                    **current_metrics,
                    **running_metrics,
                },
                tags={
                    "split": "test",
                    "type": (
                        "sample"
                        if self._scenario.test_case.data_wise == "sample_wise"
                        else "batch"
                    ),
                },
                display=False,
            )

    def _finish_domain(self) -> Dict[str, Any]:
        self._sync_cuda()
        active = self._active_domain
        descriptor = active["descriptor"]
        elapsed_seconds = time.perf_counter() - active["start_time"]
        samples = active["samples"]
        summary = {
            **descriptor,
            "processed_samples": samples,
            "processed_batches": active["batches"],
            "update_batches": active["update_batches"],
            "prediction_only_batches": active["prediction_only_batches"],
            "prediction_only_samples": active["prediction_only_samples"],
            "reset_applied": active["reset_applied"],
            "reset_reason": active["reset_reason"],
            "stream_seconds": elapsed_seconds,
            "samples_per_second": samples / elapsed_seconds
            if elapsed_seconds > 0
            else None,
        }
        summary.update(
            {
                metric_name: metric_sum / samples
                for metric_name, metric_sum in active["metric_sums"].items()
                if samples > 0
            }
        )
        for prefix_size, prefix_metrics in active["prefix_metrics"].items():
            for metric_name, metric_value in prefix_metrics.items():
                summary[f"{metric_name}_at_{prefix_size}"] = metric_value
        for prefix_size, prefix_metrics in active["batch_prefix_metrics"].items():
            for metric_name, metric_value in prefix_metrics.items():
                summary[f"{metric_name}_at_{prefix_size}_batches"] = metric_value
        summary["sample_prefix_metric_definition"] = (
            "exact_samples"
            if self._scenario.test_case.data_wise == "sample_wise"
            else "estimated_from_batch_means"
        )

        summary["memory_at_domain_end"] = self._storage_snapshot(
            include_persistent=False
        )
        self._domain_summaries.append(summary)
        self._logger.log_metric(
            name="domain",
            values={"time": time.strftime("%Y-%m-%d %H:%M:%S"), **summary},
            tags={"split": "test", "type": "domain_summary"},
            display=True,
        )
        self._active_domain = None
        return summary

    def _replay_accuracy_summaries(self) -> Dict[str, List[Dict[str, Any]]]:
        """Aggregate existing accuracy values; no additional model metric is run."""
        if not any("replay_visit" in summary for summary in self._domain_summaries):
            return {"by_visit": [], "by_corruption": []}

        def aggregate(group_key: str) -> List[Dict[str, Any]]:
            grouped = defaultdict(list)
            for summary in self._domain_summaries:
                if group_key in summary and "accuracy_top1" in summary:
                    grouped[summary[group_key]].append(summary)

            results = []
            for key in sorted(grouped, key=lambda value: (str(type(value)), value)):
                rows = grouped[key]
                total_samples = sum(row["processed_samples"] for row in rows)
                weighted_accuracy = (
                    sum(
                        row["accuracy_top1"] * row["processed_samples"]
                        for row in rows
                    )
                    / total_samples
                )
                results.append(
                    {
                        group_key: key,
                        "num_segments": len(rows),
                        "processed_samples": total_samples,
                        "accuracy_top1": weighted_accuracy,
                        "mean_segment_accuracy_top1": sum(
                            row["accuracy_top1"] for row in rows
                        )
                        / len(rows),
                    }
                )
            return results

        return {
            "by_visit": aggregate("replay_visit"),
            "by_corruption": aggregate("corruption"),
        }

    def _write_performance_summary(self, performance: Dict[str, Any]) -> None:
        artifact_sizes = {}
        for filename in (
            "arguments.json",
            "log.jsonl",
            "log-1.json",
            "log.txt",
            "latest_checkpoint.json",
        ):
            path = os.path.join(self._checkpoint_path, filename)
            if os.path.exists(path):
                artifact_sizes[filename] = os.path.getsize(path)
        performance["logging_storage_bytes"] = artifact_sizes
        performance["logging_storage_bytes_total"] = sum(artifact_sizes.values())
        performance["recovery_checkpoint_storage_bytes"] = {
            filename: os.path.getsize(os.path.join(self._checkpoint_path, filename))
            for filename in os.listdir(self._checkpoint_path)
            if filename.startswith("ctta_checkpoint_domain_")
            and filename.endswith(".pt")
        }
        performance["orphaned_log_storage_bytes"] = {
            filename: os.path.getsize(os.path.join(self._checkpoint_path, filename))
            for filename in os.listdir(self._checkpoint_path)
            if filename.startswith("orphaned-")
        }

        performance_path = os.path.join(self._checkpoint_path, "performance.json")
        with open(performance_path, "w") as fp:
            json.dump(performance, fp, indent=2)

    @property
    def _batch_size(self):
        """This function is responsible for applying the batch-wise or sample-wise setting."""
        # it may have the internal constraints for some algos.
        assert self._scenario.test_case.data_wise in ["sample_wise", "batch_wise"]
        if self._scenario.test_case.data_wise == "sample_wise":
            return 1
        elif (
            self._scenario.test_case.data_wise
            == "batch_wise"
        ):
            return self._scenario.test_case.batch_size
        else:
            raise ValueError("invalid argument in _batch_size")

    def _offline_adapt(self):
        if self._scenario.test_case.offline_pre_adapt:
            auxiliary_loader = self._model_adaptation_cls.get_auxiliary_loader(
                scenario=self._scenario
            )
            assert (
                auxiliary_loader is not None
            ), "offline_adapt needs auxiliary_loader is not None."

            self._model_adaptation_cls.offline_adapt(
                model_selection_method=self._model_selection_cls,
                auxiliary_loader=auxiliary_loader,
                timer=self._timer,
                logger=self._logger,
                random_seed=self._meta_conf.seed,
            )

    def _online_adapt_step(
        self,
        step: int,
        epoch: int,
        batch: Batch,
        previous_batches: List[Batch],
        perform_update: bool = True,
    ):
        if perform_update:
            with self._timer("adapt_and_eval", step=step, epoch=epoch):
                self._model_adaptation_cls.adapt_and_eval(
                    episodic=self._scenario.test_case.episodic,
                    metrics=self._metrics,
                    model_selection_method=self._model_selection_cls,
                    current_batch=batch,
                    previous_batches=previous_batches,
                    logger=self._logger,
                    timer=self._timer,
                )
        else:
            with self._timer("predict_without_update", step=step, epoch=epoch):
                model = self._model_adaptation_cls._model
                with auxiliary.fork_rng_with_seed(self._meta_conf.seed):
                    with torch.no_grad():
                        y_hat = model(batch._x)
                self._metrics.eval(batch._y, y_hat)

        with self._timer("data_swap", step=step, epoch=epoch):
            # previous_batches.append(batch.to(device="cpu"))
            previous_batches.append(None)
        return previous_batches

    def _with_adainit_oracle_lookahead(self, domain_iterator):
        """Attach a bounded future window only for the explicit oracle diagnostic.

        The normal benchmark never prefetches or exposes future batches to an
        adaptation method.  Keeping this path behind a selector whose name says
        ``oracle`` prevents accidental use in formal unlabeled experiments.
        """

        selector = getattr(
            self._meta_conf, "adainit_initialization_selector", "surrogate"
        )
        horizon = getattr(self._meta_conf, "adainit_oracle_horizon", 0)
        if (
            self._scenario.model_adaptation_method != "adainit"
            or selector != "oracle_future_accuracy"
            or horizon <= 0
        ):
            yield from domain_iterator
            return

        iterator = iter(domain_iterator)
        prefetched = deque(islice(iterator, horizon + 1))
        while prefetched:
            current = prefetched.popleft()
            self._model_adaptation_cls.set_oracle_lookahead(
                [item[2] for item in islice(prefetched, 0, horizon)]
            )
            yield current
            try:
                prefetched.append(next(iterator))
            except StopIteration:
                pass

    def eval(self) -> Dict:
        evaluation_start = time.perf_counter()
        self._logger.log(
            f"Test-time adaptation benchmark: scenarios={self._scenario}", display=False
        )
        self._logger.pretty_print(self._scenario)

        # safety check
        assert self._test_loader is not None

        # log dataset statistics
        if "shiftedlabel" in self._meta_conf.data_names:
            self._logger.log_metric(
                name="runtime",
                values={
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "dataset_statistics": self._test_loader.dataset.query_dataset_attr("label_statistics"),
                },
                tags={"split": "test", "type": "overall"},
                display=True,
            )

        with auxiliary.evaluation_monitor(self._meta_conf):
            # Test-time evaluation begins.
            pre_stream_start = time.perf_counter()
            if self._resumed_from is None:
                self._offline_adapt()
                if self._meta_conf.fishers:
                    self._model_adaptation_cls.compute_fishers(
                        scenario=self._scenario, data_size=self._meta_conf.fisher_size
                    )
            self._sync_cuda()
            pre_stream_setup_seconds = time.perf_counter() - pre_stream_start

            # Reconstruct only the history length used by adaptation-step schedules;
            # actual previous image tensors are intentionally not retained.
            previous_batches: List[Batch] = [
                None
            ] * self._resume_previous_batches_count
            processed_samples = self._resume_processed_samples
            last_step = self._resume_last_step
            checkpoint_io_seconds = 0.0
            self._sync_cuda()
            stream_start = time.perf_counter()

            for domain_index in range(
                self._resume_domain_index, len(self._domain_stream_datasets)
            ):
                domain_dataset = self._domain_stream_datasets[domain_index]
                self._start_domain(domain_index)
                if self._domain_schedule[domain_index]["reset_at_start"]:
                    # Adaptation-step schedules are part of the reset state too;
                    # do not let earlier 1,000-sample segments affect this one.
                    previous_batches = []
                domain_iterator = domain_dataset.iterator(
                    batch_size=self._batch_size,
                    shuffle=False,
                    repeat=False,
                    ref_num_data=None,
                    num_workers=self._meta_conf.num_workers
                    if hasattr(self._meta_conf, "num_workers")
                    else 2,
                    pin_memory=True,
                    drop_last=False,
                )

                for _, epoch, batch in self._with_adainit_oracle_lookahead(
                    domain_iterator
                ):
                    last_step += 1
                    batch_size = len(batch)
                    is_domain_partial_batch = (
                        self._scenario.test_case.data_wise == "batch_wise"
                        and batch_size < self._batch_size
                    )
                    perform_update = not is_domain_partial_batch or getattr(
                        self._meta_conf,
                        "update_on_domain_partial_batch",
                        False,
                    )

                    previous_batches = self._online_adapt_step(
                        step=last_step,
                        epoch=epoch,
                        batch=batch,
                        previous_batches=previous_batches,
                        perform_update=perform_update,
                    )
                    self._update_domain_metrics(
                        step=last_step,
                        epoch=epoch,
                        batch_size=batch_size,
                        update_applied=perform_update,
                    )
                    processed_samples += batch_size

                descriptor = self._active_domain["descriptor"]
                if self._active_domain["samples"] != descriptor["num_samples"]:
                    raise RuntimeError(
                        "The domain iterator ended before the scheduled domain was "
                        f"complete: domain={descriptor['domain_name']}, "
                        f"processed={self._active_domain['samples']}, "
                        f"expected={descriptor['num_samples']}."
                    )
                self._finish_domain()

                completed_domains = domain_index + 1
                checkpoint_interval = self._meta_conf.checkpoint_every_domains
                if (
                    checkpoint_interval > 0
                    and completed_domains % checkpoint_interval == 0
                ):
                    stream_seconds_at_checkpoint = (
                        self._resume_stream_seconds
                        + time.perf_counter()
                        - stream_start
                        - checkpoint_io_seconds
                    )
                    checkpoint_save_start = time.perf_counter()
                    self._save_recovery_checkpoint(
                        completed_domains=completed_domains,
                        processed_samples=processed_samples,
                        last_step=last_step,
                        previous_batches_count=len(previous_batches),
                        stream_seconds_completed=stream_seconds_at_checkpoint,
                    )
                    checkpoint_io_seconds += (
                        time.perf_counter() - checkpoint_save_start
                    )

            self._sync_cuda()
            stream_seconds = (
                self._resume_stream_seconds
                + time.perf_counter()
                - stream_start
                - checkpoint_io_seconds
            )

            if processed_samples != len(self._test_loader.dataset):
                raise RuntimeError(
                    "The test iterator ended before the scheduled stream was complete: "
                    f"processed={processed_samples}, expected={len(self._test_loader.dataset)}."
                )

        stats = self._metrics.tracker()
        self._logger.log(f"stats of test-time adaptation={stats}.")
        self._logger.log_metric(
            name="runtime",
            values={"time": time.strftime("%Y-%m-%d %H:%M:%S"), **stats},
            tags={"split": "test", "type": "overall"},
            display=True,
        )
        self._logger.save_json()
        replay_accuracy_summaries = self._replay_accuracy_summaries()
        performance = {
            "timing_definition": {
                "model_load_seconds": (
                    "define_model plus load_pretrained_model; for timm ImageNet models, "
                    "pretrained weights are loaded inside define_model"
                ),
                "pre_stream_setup_seconds": "offline adaptation and Fisher setup, if enabled",
                "stream_seconds": (
                    "from requesting the first test sample through completion of the last "
                    "sample; includes data loading, inference/adaptation, metric computation, "
                    "and lightweight logging"
                ),
                "per_domain_stream_seconds": (
                    "same stream interval restricted to each contiguous domain; every "
                    "domain includes its DataLoader worker startup"
                ),
            },
            "prefix_metric_definition": {
                "accuracy_top1_at_K": (
                    "first K samples; exact in sample-wise mode and estimated from "
                    "batch means in batch-wise mode because per-sample predictions "
                    "are intentionally not logged"
                ),
                "accuracy_top1_at_K_batches": (
                    "batch-wise mode only; sample-weighted accuracy over the first "
                    "K batches"
                ),
            },
            "setup_seconds": {
                "dataset": getattr(self._meta_conf, "dataset_setup_seconds", None),
                "model_load": getattr(self._meta_conf, "model_load_seconds", None),
                "adaptation_and_selection": getattr(
                    self._meta_conf,
                    "adaptation_and_selection_setup_seconds",
                    None,
                ),
                "pre_stream": pre_stream_setup_seconds,
                "checkpoint_load": getattr(
                    self._meta_conf, "checkpoint_load_seconds", None
                ),
                "checkpoint_save_current_session": checkpoint_io_seconds,
            },
            "stream_seconds": stream_seconds,
            "evaluation_seconds": time.perf_counter() - evaluation_start,
            "processed_samples": processed_samples,
            "resumed_from": self._resumed_from,
            "resume_domain_index": self._resume_domain_index,
            "record_first_n_per_domain": self._meta_conf.record_first_n_per_domain,
            "detailed_logging_granularity": (
                "first_n_samples_per_domain"
                if self._scenario.test_case.data_wise == "sample_wise"
                else "batches_intersecting_first_n_samples_per_domain"
            ),
            "domain_partial_batch_policy": (
                "update"
                if getattr(
                    self._meta_conf, "update_on_domain_partial_batch", False
                )
                else "predict_without_update"
            ),
            "domain_schedule": self._domain_schedule,
            "domain_summaries": self._domain_summaries,
            "replay_protocol": getattr(
                self._test_loader.dataset, "replay_metadata", None
            ),
            "replay_accuracy_by_visit": replay_accuracy_summaries["by_visit"],
            "replay_accuracy_by_corruption": replay_accuracy_summaries[
                "by_corruption"
            ],
            "overall_metrics": stats,
            "timer_totals_seconds": dict(self._timer.totals),
            "timer_call_counts": dict(self._timer.call_counts),
            "storage_before_stream": self._initial_storage,
            "storage_after_stream": self._storage_snapshot(),
        }
        if self._timer.cuda_available:
            performance["stream_cuda_peak_memory_allocated_bytes"] = max(
                domain_summary["memory_at_domain_end"].get(
                    "cuda_peak_memory_allocated_bytes", 0
                )
                for domain_summary in self._domain_summaries
            )
            performance["stream_cuda_peak_memory_reserved_bytes"] = max(
                domain_summary["memory_at_domain_end"].get(
                    "cuda_peak_memory_reserved_bytes", 0
                )
                for domain_summary in self._domain_summaries
            )
        self._write_performance_summary(performance)
        self._logger.close()
        return stats
