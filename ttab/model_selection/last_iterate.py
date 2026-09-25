# -*- coding: utf-8 -*-
from typing import Any, Dict

from ttab.model_selection.base_selection import BaseSelection


class LastIterate(BaseSelection):
    """Naively return the model generated from the last iterate of adaptation."""

    def __init__(self, meta_conf, model_adaptation_method):
        # Unlike oracle selection, last-iterate selection never evaluates or loads a
        # separate model. Keeping BaseSelection's deep-copied model is therefore both
        # redundant and expensive for ViT, and repeatedly calling eval() on that
        # unused CUDA copy can fail with some PyTorch/timm combinations.
        self.meta_conf = meta_conf
        self.model = None
        self.initialize()

    def initialize(self):
        self.optimal_state = None

    def clean_up(self):
        self.optimal_state = None

    def save_state(self, state, current_batch):
        self.optimal_state = state

    def select_state(self) -> Dict[str, Any]:
        """return the optimal state and sync the model defined in the model selection method."""
        return self.optimal_state

    @property
    def name(self):
        return "last_iterate"
