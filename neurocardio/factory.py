"""Model factory used by the trainer + the inference paths."""
import torch.nn as nn


def make_model(name: str, num_classes: int) -> nn.Module:
    name = name.lower()
    if name in ("snn", "neurocardio"):
        from neurocardio.model import NeuroCardio
        return NeuroCardio(num_classes=num_classes)
    if name in ("cnn", "cnn1d"):
        from neurocardio.baselines import CNN1D
        return CNN1D(num_classes=num_classes)
    if name in ("resnet", "resnet1d"):
        from neurocardio.baselines import ResNet1D
        return ResNet1D(num_classes=num_classes)
    raise ValueError(f"unknown model: {name}")
