"""ViT model construction for the released ImageNet-C experiments."""

import timm


SUPPORTED_MODELS = ("vit_base_patch16_224",)


def define_model(config):
    """Build the ImageNet-pretrained ViT used in all released experiments."""

    if config.model_name not in SUPPORTED_MODELS:
        supported = ", ".join(SUPPORTED_MODELS)
        raise ValueError(
            f"Unsupported model {config.model_name!r}. This release contains only: "
            f"{supported}."
        )

    model = timm.create_model(config.model_name, pretrained=True)
    if getattr(config, "grad_checkpoint", False):
        model.set_grad_checkpointing()
    return model


def load_pretrained_model(config, model):
    """Compatibility hook: timm loads the public ImageNet weights above."""

    return model
