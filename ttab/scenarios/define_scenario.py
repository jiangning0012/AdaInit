"""Parse explicit ImageNet-C domain names into a benchmark scenario."""

from ttab.loads.datasets.dataset_shifts import SyntheticShiftProperty
from ttab.scenarios import HomogeneousNoMixture, Scenario, TestCase, TestDomain


def _build_domain(config, data_name):
    prefix = "imagenet_c_"
    if not data_name.startswith(prefix):
        raise ValueError(
            f"Expected '<imagenet_c_><version>-<corruption>-<severity>', got {data_name!r}."
        )
    pieces = data_name[len(prefix) :].split("-", 2)
    if len(pieces) != 3:
        raise ValueError(f"Malformed ImageNet-C domain: {data_name!r}.")
    version, corruption, severity_text = pieces
    if version != "deterministic":
        raise ValueError("Only precomputed deterministic ImageNet-C is supported.")
    try:
        severity = int(severity_text)
    except ValueError as error:
        raise ValueError(f"Invalid ImageNet-C severity in {data_name!r}.") from error
    if severity not in range(1, 6):
        raise ValueError(f"ImageNet-C severity must be 1..5, got {severity}.")

    return TestDomain(
        base_data_name="imagenet",
        data_name=data_name,
        shift_type="synthetic",
        shift_property=SyntheticShiftProperty(
            shift_degree=severity,
            shift_name=corruption,
            version=version,
        ),
        domain_sampling_name=config.domain_sampling_name,
        domain_sampling_ratio=config.domain_sampling_ratio,
    )


def get_scenario(config):
    if config.test_scenario is not None:
        raise ValueError("Named scenario presets are not part of this release.")
    test_domains = [
        _build_domain(config, data_name)
        for data_name in config.data_names.split(";")
        if data_name
    ]
    if not test_domains:
        raise ValueError("At least one ImageNet-C domain is required.")

    test_case = TestCase(
        inter_domain=HomogeneousNoMixture(),
        batch_size=config.batch_size,
        data_wise=config.data_wise,
        offline_pre_adapt=config.offline_pre_adapt,
        episodic=config.episodic,
        intra_domain_shuffle=config.intra_domain_shuffle,
    )
    return Scenario(
        base_data_name=config.base_data_name,
        src_data_name=config.src_data_name,
        test_domains=test_domains,
        test_case=test_case,
        task=config.task,
        model_name=config.model_name,
        model_adaptation_method=config.model_adaptation_method,
        model_selection_method=config.model_selection_method,
    )


def _as_plain_value(value):
    if hasattr(value, "_asdict"):
        return {key: _as_plain_value(item) for key, item in value._asdict().items()}
    if isinstance(value, list):
        return [_as_plain_value(item) for item in value]
    return value


def scenario_registry(config, scenario):
    for field_name, value in scenario._asdict().items():
        setattr(config, field_name, _as_plain_value(value))
    return config
