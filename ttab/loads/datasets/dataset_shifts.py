"""Scenario metadata for deterministic ImageNet-C shifts."""

from typing import NamedTuple


class SyntheticShiftProperty(NamedTuple):
    shift_degree: int
    shift_name: str
    version: str = "deterministic"
    has_shift: bool = True
