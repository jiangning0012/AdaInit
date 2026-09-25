# -*- coding: utf-8 -*-
import time

import parameters
import torch
import ttab.configs.utils as configs_utils
import ttab.loads.define_dataset as define_dataset
from ttab.benchmark import Benchmark
from ttab.loads.define_model import define_model, load_pretrained_model
from ttab.model_adaptation import get_model_adaptation_method
from ttab.model_selection import get_model_selection_method


def _synchronize_if_cuda(device):
    if "cuda" in str(device) and torch.cuda.is_available():
        torch.cuda.synchronize(device=device)


def main(init_config):
    # Required auguments.
    config, scenario = configs_utils.config_hparams(config=init_config)

    dataset_setup_start = time.perf_counter()
    test_data_cls = define_dataset.ConstructTestDataset(config=config)
    test_loader = test_data_cls.construct_test_loader(scenario=scenario)
    config.dataset_setup_seconds = time.perf_counter() - dataset_setup_start

    # Model.
    _synchronize_if_cuda(config.device)
    model_load_start = time.perf_counter()
    model = define_model(config=config)
    load_pretrained_model(config=config, model=model)
    _synchronize_if_cuda(config.device)
    config.model_load_seconds = time.perf_counter() - model_load_start

    # Algorithms.
    _synchronize_if_cuda(config.device)
    adaptation_setup_start = time.perf_counter()
    model_adaptation_cls = get_model_adaptation_method(
        adaptation_name=scenario.model_adaptation_method
    )(meta_conf=config, model=model)
    model_selection_cls = get_model_selection_method(selection_name=scenario.model_selection_method)(
        meta_conf=config, model_adaptation_method=model_adaptation_cls
    )
    _synchronize_if_cuda(config.device)
    config.adaptation_and_selection_setup_seconds = (
        time.perf_counter() - adaptation_setup_start
    )

    # Evaluate.
    benchmark = Benchmark(
        scenario=scenario,
        model_adaptation_cls=model_adaptation_cls,
        model_selection_cls=model_selection_cls,
        test_loader=test_loader,
        meta_conf=config,
    )
    benchmark.eval()


if __name__ == "__main__":
    conf = parameters.get_args()
    main(init_config=conf)
