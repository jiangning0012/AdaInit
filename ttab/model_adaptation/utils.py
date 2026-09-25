"""Shared optimization utilities for the released adaptation methods."""

import copy

import torch


def define_optimizer(meta_conf, params, lr=1e-3):
    weight_decay = getattr(meta_conf, "weight_decay", 0.0)
    optimizer = getattr(meta_conf, "optimizer", "SGD")
    if optimizer == "SGD":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=getattr(meta_conf, "momentum", 0.9),
            dampening=getattr(meta_conf, "dampening", 0.0),
            weight_decay=weight_decay,
            nesterov=getattr(meta_conf, "nesterov", True),
        )
    if optimizer == "Adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=(getattr(meta_conf, "beta", 0.9), 0.999),
            weight_decay=weight_decay,
        )
    if optimizer == "AdamW":
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=(getattr(meta_conf, "beta", 0.9), 0.999),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer}")


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer used by SAR and SAR2."""

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        if rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}.")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                self.state[parameter]["old_p"] = parameter.data.clone()
                perturbation = (
                    (parameter.pow(2) if group["adaptive"] else 1.0)
                    * parameter.grad
                    * scale.to(parameter)
                )
                parameter.add_(perturbation)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.data = self.state[parameter]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("SAM requires a closure.")
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            ((parameter.abs() if group["adaptive"] else 1.0) * parameter.grad)
            .norm(p=2)
            .to(shared_device)
            for group in self.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if not norms:
            return torch.zeros((), device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    def state_dict(self):
        state = super().state_dict()
        if getattr(self, "base_optimizer", None) is not None:
            state["base_optimizer"] = self.base_optimizer.state_dict()
        return state

    def load_state_dict(self, state_dict):
        state_dict = copy.deepcopy(state_dict)
        base_state = state_dict.pop("base_optimizer", None)
        super().load_state_dict(state_dict)
        if getattr(self, "base_optimizer", None) is not None:
            self.base_optimizer.param_groups = self.param_groups
            if base_state is not None:
                self.base_optimizer.load_state_dict(base_state)
                self.param_groups = self.base_optimizer.param_groups


@torch.jit.script
def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


@torch.jit.script
def teacher_student_softmax_entropy(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor
) -> torch.Tensor:
    return -(teacher_logits.softmax(1) * student_logits.log_softmax(1)).sum(1)
