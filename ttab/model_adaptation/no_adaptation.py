# -*- coding: utf-8 -*-
import copy
from typing import List

import torch
import torch.nn as nn
from ttab.api import Batch
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer


class NoAdaptation(BaseAdaptation):
    """Standard test-time evaluation (no adaptation)."""

    def __init__(self, meta_conf, model: nn.Module):
        super().__init__(meta_conf, model)

    def _initialize_model(self, model: nn.Module):
        """Configure model for adaptation."""
        model.eval()
        return model.to(self._meta_conf.device)

    def _post_safety_check(self):
        pass

    def initialize(self, seed: int):
        """Initialize the algorithm."""
        self._model = self._initialize_model(model=copy.deepcopy(self._base_model))

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
        """The key entry of test-time adaptation."""
        # some simple initialization.
        with timer("test_time_adaptation"):
            with torch.no_grad():
                y_hat = self._model(current_batch._x)

        with timer("evaluate_adaptation_result"):
            metrics.eval(current_batch._y, y_hat)

    @property
    def name(self):
        return "no_adaptation"
