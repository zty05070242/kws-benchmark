"""Measure model-only CPU latency with synthetic or Speech Commands inputs.

TC-ResNet8 classifier port: hyperconnect/TC-ResNet at 8ccbff3,
audio_nets/tc_resnet.py (Apache-2.0). The auxiliary range head is unused.
BC-ResNet-1: Qualcomm-AI-research/bcresnet at 656f947, unchanged.
"""
import json
import platform
import sys
from argparse import ArgumentParser
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent / "bcresnet"))
from bcresnet import BCResNets


def temporal_layer(inputs, outputs, kernel, stride=1, relu=True):
    # TensorFlow SAME puts the extra padding on the right. The strided
    # layer defaults to even-length padding. TCBlock adjusts for odd lengths.
    layers = []
    if stride == 2 and kernel > 1:
        layers.append(nn.ConstantPad1d((kernel // 2 - 1, kernel // 2), 0))
    padding = kernel // 2 if stride == 1 else 0
    layers.extend([
        nn.Conv1d(inputs, outputs, kernel, stride, padding, bias=False),
        nn.BatchNorm1d(outputs),
    ])
    if relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class TCBlock(nn.Module):
    def __init__(self, inputs, outputs, frames):
        super().__init__()
        # For odd lengths, TensorFlow SAME pads the 9-tap strided layer
        # symmetrically. The 1-tap shortcut requires no padding in either case.
        self.first = temporal_layer(inputs, outputs, 9, stride=2)
        if frames % 2:
            self.first[0] = nn.ConstantPad1d((4, 4), 0)
        self.second = temporal_layer(outputs, outputs, 9, relu=False)
        self.shortcut = temporal_layer(inputs, outputs, 1, stride=2)

    def forward(self, x):
        return (self.second(self.first(x)) + self.shortcut(x)).relu()


def measure_graph(model, features):
    macs = 0

    def count(layer, _inputs, output):
        nonlocal macs
        macs += output.numel() * layer.weight[0].numel()

    hooks = [
        layer.register_forward_hook(count)
        for layer in model.modules()
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear))
    ]
    output = model(features)
    assert output.shape == (1, 12) and torch.isfinite(output).all()
    for hook in hooks:
        hook.remove()

    params = list(model.parameters())
    return {
        "input_shape": list(features.shape),
        "parameters": sum(p.numel() for p in params),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in params),
        "conv_and_linear_macs": macs,
    }


def tc_resnet8():
    return nn.Sequential(
        temporal_layer(40, 16, 3),
        TCBlock(16, 24, 98),
        TCBlock(24, 32, 49),
        TCBlock(32, 48, 25),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(48, 12, bias=False),
    )


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--extensions", action="store_true")
    parser.add_argument("--esc50-seed", type=int, help="Use trained checkpoints and the same test clip.")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    models = {
        "TC-ResNet8 (PyTorch port)": (tc_resnet8().eval(), torch.randn(1, 40, 98)),
        "BC-ResNet-1 (released code)": (
            BCResNets(8).eval(), torch.randn(1, 1, 40, 101)
        ),
    }
    if args.extensions:
        from extensions import ds_cnn_small, matchboxnet

        models["DS-CNN-S (PyTorch port)"] = (ds_cnn_small().eval(), torch.randn(1, 1, 49, 10))
        models["MatchboxNet-3x2x64 (PyTorch port)"] = (matchboxnet().eval(), torch.randn(1, 64, 128))
    if args.esc50_seed is not None:
        root = Path(__file__).resolve().parent
        for name, key in zip(models, ("tc", "bc", "ds", "matchbox")):
            model, _ = models[name]
            weights = torch.load(root / f"{key}-best-{args.esc50_seed}-esc50.pt",
                                 map_location="cpu", weights_only=True)
            model.load_state_dict(weights)
            features, _ = torch.load(root / "data" / f"{key}-test-features.pt",
                                     map_location="cpu", weights_only=True)
            models[name] = (model, features[:1].clone())
    timings = {name: [] for name in models}
    with torch.inference_mode():
        results = {
            name: measure_graph(model, features)
            for name, (model, features) in models.items()
        }
        for model, features in models.values():
            for _ in range(100):
                model(features)
        for trial in range(8):
            order = list(models) if trial % 2 == 0 else list(reversed(models))
            for name in order:
                model, features = models[name]
                start = perf_counter_ns()
                for _ in range(250):
                    model(features)
                timings[name].append((perf_counter_ns() - start) / 250 / 1e6)
    for name, samples in timings.items():
        results[name]["median_model_ms"] = median(samples)
        results[name]["mean_ms_per_trial"] = samples
    print(json.dumps({
        "platform": platform.platform(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "precision": "float32",
        "esc50_checkpoint_seed": args.esc50_seed,
        "input": "synthetic" if args.esc50_seed is None else "first official v2 test clip",
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
