"""Dataset constants for the released ImageNet-C benchmark."""

dataset_defaults = {
    "imagenet": {
        "statistics": {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "n_classes": 1000,
        },
        "img_shape": (224, 224, 3),
    }
}
