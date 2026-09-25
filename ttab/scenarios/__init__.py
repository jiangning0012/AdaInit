"""Scenario types used by the released continual ImageNet-C streams."""

from typing import List, NamedTuple

from ttab.loads.datasets.dataset_shifts import SyntheticShiftProperty


class TestDomain(NamedTuple):
    base_data_name: str
    data_name: str
    shift_type: str
    shift_property: SyntheticShiftProperty
    domain_sampling_name: str = "uniform"
    domain_sampling_value: float = None
    domain_sampling_ratio: float = 1.0


class HomogeneousNoMixture(NamedTuple):
    has_mixture: bool = False


class TestCase(NamedTuple):
    inter_domain: HomogeneousNoMixture
    batch_size: int = 1
    data_wise: str = "sample_wise"
    offline_pre_adapt: bool = False
    episodic: bool = False
    intra_domain_shuffle: bool = True


class Scenario(NamedTuple):
    task: str
    model_name: str
    model_adaptation_method: str
    model_selection_method: str
    base_data_name: str
    src_data_name: str
    test_domains: List[TestDomain]
    test_case: TestCase
