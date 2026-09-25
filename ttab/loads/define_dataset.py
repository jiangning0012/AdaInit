"""Construct the two released continual ImageNet-C streams."""

import numpy as np
import torch

from ttab.api import PyTorchDataset
from ttab.loads.datasets import loaders
from ttab.loads.datasets.dataset_sampling import DatasetSampling
from ttab.loads.datasets.datasets import ImageNetCDataset
from ttab.scenarios import HomogeneousNoMixture


class ConstructTestDataset:
    def __init__(self, config):
        self.meta_conf = config

    @staticmethod
    def _parse_domain(test_domain):
        if test_domain.base_data_name != "imagenet":
            raise ValueError("This release supports ImageNet-C only.")
        if test_domain.shift_type != "synthetic":
            raise ValueError(f"Expected an ImageNet-C domain, got {test_domain.data_name!r}.")
        if test_domain.shift_property.version != "deterministic":
            raise ValueError("Only precomputed deterministic ImageNet-C is supported.")
        return (
            test_domain.shift_property.shift_name,
            test_domain.shift_property.shift_degree,
        )

    def get_test_datasets(self, test_domains):
        datasets = []
        for test_domain in test_domains:
            corruption, severity = self._parse_domain(test_domain)
            dataset = ImageNetCDataset(
                root=self.meta_conf.data_path,
                corruption=corruption,
                severity=severity,
                device=self.meta_conf.device,
            )
            datasets.append(
                DatasetSampling(test_domain).sample(
                    dataset, random_seed=self.meta_conf.seed
                )
            )
        return datasets

    @staticmethod
    def _stream_from_children(children, replay_metadata=None):
        stream = PyTorchDataset(
            dataset=torch.utils.data.ConcatDataset(
                [child.dataset for child in children]
            ),
            device=children[0]._device,
            prepare_batch=children[0]._prepare_batch,
            num_classes=children[0].num_classes,
        )
        # Benchmark consumes children independently so exact domain boundaries
        # remain observable for reporting and optional baseline resets.
        stream.datasets = children
        if replay_metadata is not None:
            stream.replay_metadata = replay_metadata
        return stream

    def _construct_round_major_replay(self, scenario, test_datasets):
        visits = int(self.meta_conf.domain_replay_visits)
        samples_per_visit = int(self.meta_conf.domain_replay_samples_per_visit)
        if visits <= 1 or samples_per_visit <= 0:
            raise ValueError("Replay requires visits > 1 and samples_per_visit > 0.")
        if not isinstance(scenario.test_case.inter_domain, HomogeneousNoMixture):
            raise ValueError("Replay requires HomogeneousNoMixture.")

        required = visits * samples_per_visit
        for index, dataset in enumerate(test_datasets):
            if len(dataset) != required:
                raise ValueError(
                    f"Domain {index} has {len(dataset)} sampled images; replay needs "
                    f"{visits} x {samples_per_visit} = {required}."
                )

        positions = (
            np.random.default_rng(self.meta_conf.seed).permutation(required).tolist()
            if scenario.test_case.intra_domain_shuffle
            else list(range(required))
        )
        children = []
        for visit in range(visits):
            start = visit * samples_per_visit
            end = start + samples_per_visit
            for domain_index, dataset in enumerate(test_datasets):
                subset = torch.utils.data.Subset(dataset.dataset, positions[start:end])
                child = PyTorchDataset(
                    dataset=subset,
                    device=dataset._device,
                    prepare_batch=dataset._prepare_batch,
                    num_classes=dataset.num_classes,
                )
                child.transform = dataset.transform
                child.target_transform = dataset.target_transform
                child.replay_metadata = {
                    "base_domain_index": domain_index,
                    "replay_visit": visit + 1,
                    "domain_within_visit": domain_index + 1,
                    "num_replay_visits": visits,
                    "samples_per_visit": samples_per_visit,
                    "sample_partition_start": start,
                    "sample_partition_end": end - 1,
                }
                children.append(child)

        return self._stream_from_children(
            children,
            replay_metadata={
                "order": "round_major",
                "num_replay_visits": visits,
                "num_base_domains": len(test_datasets),
                "samples_per_visit": samples_per_visit,
                "aligned_source_image_partitions": True,
                "seed": self.meta_conf.seed,
            },
        )

    def construct_test_dataset(self, scenario, data_augment=False):
        if data_augment:
            raise ValueError("The released evaluation stream does not augment inputs.")
        datasets = self.get_test_datasets(scenario.test_domains)
        if int(self.meta_conf.domain_replay_visits) > 1:
            return self._construct_round_major_replay(scenario, datasets)

        if scenario.test_case.intra_domain_shuffle:
            for dataset in datasets:
                dataset.replace_indices(
                    indices_pattern="random_shuffle",
                    random_seed=self.meta_conf.seed,
                )
        return self._stream_from_children(datasets)

    def construct_test_loader(self, scenario):
        return loaders.get_test_loader(
            self.construct_test_dataset(scenario), device=self.meta_conf.device
        )


class ConstructAuxiliaryDataset(ConstructTestDataset):
    """Compatibility path for TTAB; not used by the released no-Fisher runs."""

    def construct_auxiliary_loader(self, scenario, data_augment=False):
        return loaders.get_auxiliary_loader(
            self.construct_test_dataset(scenario, data_augment=False),
            device=self.meta_conf.device,
        )

    def construct_src_dataset(self, scenario, data_size, data_augment=False):
        raise RuntimeError("Source-data Fisher estimation is not part of this release.")
