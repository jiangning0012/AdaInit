"""Dataset wrappers required by the released ImageNet-C protocols."""

from pathlib import Path

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from ttab.api import Batch, PyTorchDataset


class IndexedImageFolder(ImageFolder):
    """ImageFolder with an explicit, replaceable sample order."""

    def __init__(self, root, transform=None):
        super().__init__(root=str(root), transform=transform)
        self.indices = list(range(len(self.samples)))
        self.data_size = len(self.indices)
        self.data = [path for path, _ in self.samples]
        self.class_to_index = self.class_to_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return super().__getitem__(self.indices[index])

    def update_indices(self, new_indices):
        self.indices = list(new_indices)
        self.data_size = len(self.indices)


class ImageNetCDataset(PyTorchDataset):
    """One corruption/severity domain from an extracted ImageNet-C tree."""

    def __init__(self, root, corruption, severity, device="cuda"):
        domain_root = Path(root) / "ILSVRC" / "imagenet-c" / corruption / str(severity)
        if not domain_root.is_dir():
            raise FileNotFoundError(
                f"ImageNet-C domain not found: {domain_root}. See README.md for the "
                "required directory layout."
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self.target_transform = None
        dataset = IndexedImageFolder(domain_root, transform=self.transform)
        super().__init__(
            dataset=dataset,
            device=device,
            prepare_batch=self.prepare_batch,
            num_classes=1000,
        )

    @staticmethod
    def prepare_batch(batch, device):
        return Batch(*batch).to(device)


class WrapperDataset(PyTorchDataset):
    """Wrap a native PyTorch dataset in TTAB's iterator interface."""

    def __init__(self, dataset, device="cuda"):
        super().__init__(dataset, device, self.prepare_batch, num_classes=None)

    @staticmethod
    def prepare_batch(batch, device):
        return Batch(*batch).to(device)
