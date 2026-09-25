# -*- coding: utf-8 -*-
"""Backend adaptation rules used by AdaInit.

AdaInit changes where online adaptation starts; it does not replace the loss or
optimizer of the wrapped TTA method.  These small adapters keep that boundary
explicit and let the same initialization framework run with Tent, CoME, SAR,
AdaDEM, or NCTTA.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import ttab.model_adaptation.utils as adaptation_utils
from ttab.api import Batch
from ttab.model_adaptation.nctta import compute_nc_loss
from ttab.utils.auxiliary import fork_rng_with_seed
from ttab.utils.timer import Timer


@dataclass
class BackendStepResult:
    yhat: torch.Tensor
    loss: float
    reset_requested: bool = False
    surrogate_loss: Optional[float] = None


class AdaInitBackend:
    """Backend interface shared by formal and counterfactual updates."""

    name = "base"
    # Stateful backends can request that every hard initialization candidate
    # starts from the same copy of the current online statistic.  This keeps
    # candidate comparisons about model initialization rather than hidden
    # backend-state differences.
    share_runtime_state_across_candidates = False
    # AdaInit caches and selects model parameters only.  A stateful backend can
    # preserve its live statistic when a non-current initialization is chosen.
    preserve_runtime_state_on_hard_initialization = False

    def configure_model(self, model: nn.Module, device: str) -> nn.Module:
        model.train()
        model.requires_grad_(False)
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.requires_grad_(True)
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
            if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                module.requires_grad_(True)
        return model.to(device)

    def collect_parameters(
        self, model: nn.Module
    ) -> Tuple[List[nn.Parameter], List[str], List[str]]:
        module_names = []
        parameters = []
        parameter_names = []
        for module_name, module in model.named_modules():
            if not self.adapt_module(module_name, module):
                continue
            module_names.append(module_name)
            for parameter_name, parameter in module.named_parameters(recurse=False):
                if parameter_name in {"weight", "bias"}:
                    parameters.append(parameter)
                    parameter_names.append(f"{module_name}.{parameter_name}")
        if not parameters:
            raise RuntimeError(
                f"AdaInit+{self.name} needs adaptable normalization parameters."
            )
        return parameters, parameter_names, module_names

    def adapt_module(self, module_name: str, module: nn.Module) -> bool:
        return isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm))

    def build_optimizer(self, meta_conf, params) -> torch.optim.Optimizer:
        return adaptation_utils.define_optimizer(meta_conf, params, lr=meta_conf.lr)

    def fresh_runtime_state(self) -> Dict[str, Any]:
        return {}

    def adapt_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        runtime_state: Dict[str, Any],
        timer: Timer,
        random_seed: int = None,
    ) -> BackendStepResult:
        raise NotImplementedError


class TentBackend(AdaInitBackend):
    name = "tent"

    def adapt_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        runtime_state: Dict[str, Any],
        timer: Timer,
        random_seed: int = None,
    ) -> BackendStepResult:
        optimizer.zero_grad()
        with timer("adainit.backend_forward"):
            with fork_rng_with_seed(random_seed):
                yhat = model(batch._x)
            loss = adaptation_utils.softmax_entropy(yhat).mean()
        with timer("adainit.backend_backward"):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return BackendStepResult(yhat=yhat, loss=float(loss.detach().item()))


class COMEBackend(AdaInitBackend):
    name = "come"

    @staticmethod
    def entropy_of_opinion(logits: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(logits, p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        normalized_logits = logits / norm * norm.detach()
        number_of_classes = logits.shape[-1]
        exp_logits = torch.exp(normalized_logits)
        strength = exp_logits.sum(dim=1, keepdim=True) + number_of_classes
        belief = exp_logits / strength
        uncertainty = number_of_classes / strength
        opinion = torch.cat([belief, uncertainty], dim=1).clamp_min(1e-7)
        return -(opinion * torch.log(opinion)).sum(dim=1)

    def adapt_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        runtime_state: Dict[str, Any],
        timer: Timer,
        random_seed: int = None,
    ) -> BackendStepResult:
        optimizer.zero_grad()
        with timer("adainit.backend_forward"):
            with fork_rng_with_seed(random_seed):
                yhat = model(batch._x)
            loss = self.entropy_of_opinion(yhat).mean()
        with timer("adainit.backend_backward"):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return BackendStepResult(yhat=yhat, loss=float(loss.detach().item()))


class AdaDEMBackend(AdaInitBackend):
    """Faithful AdaDEM update with a branch-local class-marginal estimator.

    ``avg_pred`` is online runtime state, not learned model knowledge.  It is
    copied equally to all counterfactual candidates and is deliberately absent
    from AdaInit's lightweight parameter cache.
    """

    name = "adadem"
    share_runtime_state_across_candidates = True
    preserve_runtime_state_on_hard_initialization = True

    def fresh_runtime_state(self) -> Dict[str, Any]:
        return {"avg_pred": None}

    def _compute_loss(
        self, logits: torch.Tensor, runtime_state: Dict[str, Any]
    ) -> torch.Tensor:
        probabilities = F.softmax(logits, dim=1)
        pseudo_labels = probabilities.argmax(dim=1)

        with torch.no_grad():
            average_prediction = runtime_state.get("avg_pred")
            if (
                average_prediction is None
                or average_prediction.shape
                != (logits.shape[1], logits.shape[1])
            ):
                average_prediction = torch.ones(
                    (logits.shape[1], logits.shape[1]),
                    device=logits.device,
                    dtype=probabilities.dtype,
                ) / logits.shape[1]
            else:
                average_prediction = average_prediction.to(
                    device=logits.device, dtype=probabilities.dtype
                ).detach()
            for predicted_class in torch.unique(pseudo_labels):
                class_mask = pseudo_labels == predicted_class
                average_prediction[predicted_class] = (
                    (1.0 - self.meta_conf.adadem_pi)
                    * average_prediction[predicted_class]
                    + self.meta_conf.adadem_pi
                    * probabilities[class_mask].mean(dim=0).detach()
                )
            runtime_state["avg_pred"] = average_prediction.detach()

            expected_logit = -(
                probabilities * logits
            ).sum(dim=1, keepdim=True)
            gradient = (logits + expected_logit + 1.0) * probabilities
            gradient_norm = gradient.abs().sum(dim=1, keepdim=True)

        mode = self.meta_conf.adadem_mode
        if mode == "adadem":
            adjusted_probability = (
                probabilities
                - runtime_state["avg_pred"][pseudo_labels].detach()
            ) / gradient_norm.detach()
        elif mode == "adadem-norm":
            adjusted_probability = (
                probabilities - probabilities.detach()
            ) / gradient_norm.detach()
        elif mode == "adadem-mec":
            adjusted_probability = (
                probabilities
                - runtime_state["avg_pred"][pseudo_labels].detach()
            )
        else:
            raise ValueError(f"Unknown AdaDEM mode: {mode}")
        return -(adjusted_probability * logits).sum(dim=1).mean(dim=0)

    def adapt_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        runtime_state: Dict[str, Any],
        timer: Timer,
        random_seed: int = None,
    ) -> BackendStepResult:
        optimizer.zero_grad()
        with timer("adainit.backend_forward"):
            with fork_rng_with_seed(random_seed):
                yhat = model(batch._x)
            loss = self._compute_loss(yhat, runtime_state)
        with timer("adainit.backend_backward"):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return BackendStepResult(yhat=yhat, loss=float(loss.detach().item()))


class NCTTABackend(AdaInitBackend):
    """Faithful NCTTA update rule with its classifier-geometry objective."""

    name = "nctta"

    def configure_model(self, model: nn.Module, device: str) -> nn.Module:
        model = super().configure_model(model, device)
        feature_layer = self._get_feature_layer(model)
        self._cached_features = None
        # Keep the handle alive for the lifetime of the backend. The hook is on
        # the formal AdaInit model and is also used by temporary branches.
        self._feature_hook_handle = feature_layer.register_forward_hook(
            self._feature_hook
        )
        self._classifier_weight = self._get_classifier_weight(model)
        return model

    @staticmethod
    def _get_feature_layer(model: nn.Module) -> nn.Module:
        for name in ("fc_norm", "global_pool", "avgpool"):
            layer = getattr(model, name, None)
            if layer is not None:
                return layer
        raise RuntimeError(
            "AdaInit+NCTTA cannot find fc_norm, global_pool, or avgpool."
        )

    @staticmethod
    def _get_classifier_weight(model: nn.Module) -> torch.Tensor:
        for name in ("classifier", "fc", "head"):
            classifier = getattr(model, name, None)
            if classifier is not None and hasattr(classifier, "weight"):
                return classifier.weight.detach()
        raise RuntimeError("AdaInit+NCTTA cannot find the classifier weight.")

    def _feature_hook(self, module, inputs, output) -> None:
        features = output.reshape(output.shape[0], -1)
        self._cached_features = F.normalize(features, p=2, dim=-1)

    def _compute_loss(
        self, logits: torch.Tensor, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normalized_features = F.normalize(features, p=2, dim=1)
        normalized_weights = F.normalize(self._classifier_weight, p=2, dim=1)
        predicted_classes = torch.argmax(logits, dim=1)
        entropy = adaptation_utils.softmax_entropy(logits)
        nc_loss, _ = compute_nc_loss(
            weight=self._classifier_weight,
            features=features,
            y_hat=logits,
            top_k=self.meta_conf.top_k,
            type="infonce",
            metric="cos",
            tau_align=1.0,
            margin=0.2,
            mix_prob_weight=self.meta_conf.mix_prob_weight,
            y_hat_is_logits=True,
            reduce="none",
        )
        selected_weights = normalized_weights[predicted_classes]
        distance = torch.norm(
            normalized_features - selected_weights, p=2, dim=1
        )
        entropy_coefficient = self.meta_conf.reweight_ent / torch.exp(
            entropy.detach() - self.meta_conf.margin_ent
        )
        distance_coefficient = self.meta_conf.nu / (
            1.0 + self.meta_conf.eta * distance.detach()
        )
        per_sample_loss = (entropy + self.meta_conf.scale * nc_loss) * (
            entropy_coefficient + distance_coefficient
        )
        optimization_loss = per_sample_loss[
            entropy < self.meta_conf.thre_ent
        ].mean(0)
        # NCTTA intentionally filters unreliable samples from its formal update.
        # With batch size one that set can be empty, so the optimization loss is
        # NaN.  Keep that native behavior, but expose the same objective before
        # filtering as a finite, unlabeled candidate-selection diagnostic.
        surrogate_loss = per_sample_loss.mean(0)
        return optimization_loss, surrogate_loss

    def adapt_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        runtime_state: Dict[str, Any],
        timer: Timer,
        random_seed: int = None,
    ) -> BackendStepResult:
        optimizer.zero_grad()
        with timer("adainit.backend_forward"):
            self._cached_features = None
            with fork_rng_with_seed(random_seed):
                yhat = model(batch._x)
            if self._cached_features is None:
                raise RuntimeError("AdaInit+NCTTA feature hook captured no features.")
            loss, surrogate_loss = self._compute_loss(
                yhat, self._cached_features
            )
        with timer("adainit.backend_backward"):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return BackendStepResult(
            yhat=yhat,
            loss=float(loss.detach().item()),
            surrogate_loss=float(surrogate_loss.detach().item()),
        )


class SARBackend(AdaInitBackend):
    name = "sar"

    def adapt_module(self, module_name: str, module: nn.Module) -> bool:
        return "layer4" not in module_name and super().adapt_module(
            module_name, module
        )

    def build_optimizer(self, meta_conf, params) -> torch.optim.Optimizer:
        return adaptation_utils.SAM(
            params,
            base_optimizer=torch.optim.SGD,
            lr=meta_conf.lr,
            momentum=getattr(meta_conf, "momentum", 0.9),
        )

    def fresh_runtime_state(self) -> Dict[str, Any]:
        return {"ema": None}

    @staticmethod
    def _update_ema(previous, current: float) -> float:
        return current if previous is None else 0.9 * previous + 0.1 * current

    def adapt_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        runtime_state: Dict[str, Any],
        timer: Timer,
        random_seed: int = None,
    ) -> BackendStepResult:
        margin = self.meta_conf.sar_margin_e0
        optimizer.zero_grad()
        with timer("adainit.backend_forward"):
            with fork_rng_with_seed(random_seed):
                yhat = model(batch._x)
            entropy = adaptation_utils.softmax_entropy(yhat)
            reliable = entropy < margin
            if not torch.any(reliable):
                # Native SAR still executes both SAM steps for an empty reliable
                # set: the gradients are zero, but SGD momentum is advanced. Use
                # explicit zero losses to preserve that state transition without
                # relying on backward() from a NaN empty mean.
                zero_first_loss = yhat.sum() * 0.0
                zero_first_loss.backward()
                optimizer.first_step(zero_grad=True)
                zero_second_loss = model(batch._x).sum() * 0.0
                zero_second_loss.backward()
                optimizer.second_step(zero_grad=True)
                return BackendStepResult(yhat=yhat, loss=float("nan"))
            first_loss = entropy[reliable].mean()

        with timer("adainit.backend_backward"):
            first_loss.backward()
            optimizer.first_step(zero_grad=True)

            second_entropy = adaptation_utils.softmax_entropy(model(batch._x))
            second_entropy = second_entropy[reliable]
            second_reliable = second_entropy < margin
            if not torch.any(second_reliable):
                # As above, native SAR completes SAM with zero gradients and can
                # therefore advance the base optimizer's momentum.
                (second_entropy.sum() * 0.0).backward()
                optimizer.second_step(zero_grad=True)
                return BackendStepResult(
                    yhat=yhat, loss=float(first_loss.detach().item())
                )

            second_loss = second_entropy[second_reliable].mean()
            second_loss.backward()
            optimizer.second_step(zero_grad=True)

        runtime_state["ema"] = self._update_ema(
            runtime_state.get("ema"), float(second_loss.detach().item())
        )
        reset_requested = runtime_state["ema"] < self.meta_conf.reset_constant_em
        return BackendStepResult(
            yhat=yhat,
            loss=float(first_loss.detach().item()),
            reset_requested=reset_requested,
        )


def build_adainit_backend(meta_conf) -> AdaInitBackend:
    backend_name = meta_conf.adainit_backend.lower()
    backend_classes = {
        "tent": TentBackend,
        "come": COMEBackend,
        "sar": SARBackend,
        "adadem": AdaDEMBackend,
        "nctta": NCTTABackend,
    }
    if backend_name not in backend_classes:
        raise ValueError(f"Unsupported AdaInit backend: {backend_name}")
    backend = backend_classes[backend_name]()
    # Keep configuration on the adapter without making it a torch module.
    backend.meta_conf = meta_conf
    return backend
