"""Trace the CPU latency reversal between two trained keyword spotters."""

import json
import platform
from argparse import ArgumentParser
from collections import Counter
from copy import deepcopy
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import torch
from torch import nn
from torch.nn import functional as F
from torch.profiler import ProfilerActivity, profile

from benchmark import BCResNets, measure_graph
from extensions import ds_cnn_small


class AxisDepthwise(nn.Module):
    """Evaluate the same three-tap filter with shifted views of the input."""

    def __init__(self, layer):
        super().__init__()
        assert layer.groups == layer.in_channels == layer.out_channels
        assert layer.bias is None and layer.kernel_size in ((3, 1), (1, 3))
        height, width = layer.padding
        self.padding = (width, width, height, height)
        axis = layer.kernel_size.index(3)
        self.axis = axis + 2
        self.stride = layer.stride[axis]
        self.dilation = layer.dilation[axis]
        taps = layer.weight.detach().reshape(layer.out_channels, 3)
        for index, tap in enumerate(taps.unbind(1)):
            self.register_buffer(f"tap{index}", tap.reshape(1, -1, 1, 1).contiguous())

    def forward(self, x):
        windows = F.pad(x, self.padding).unfold(self.axis, 2 * self.dilation + 1, self.stride)
        y = windows[..., 0] * self.tap0
        y = torch.addcmul(y, windows[..., self.dilation], self.tap1)
        return torch.addcmul(y, windows[..., 2 * self.dilation], self.tap2)


def replace_axes(model, kernel_sizes):
    variant = deepcopy(model)
    for name, layer in list(variant.named_modules()):
        if isinstance(layer, nn.Conv2d) and layer.kernel_size in kernel_sizes:
            variant.set_submodule(name, AxisDepthwise(layer))
    return variant.eval()


def load_models():
    root = Path(__file__).resolve().parent
    cases = {}
    for name, model in (("bc", BCResNets(8)), ("ds", ds_cnn_small())):
        weights = torch.load(root / f"{name}-best-42-esc50.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(weights)
        features, labels = torch.load(
            root / "data" / f"{name}-test-features.pt", map_location="cpu", weights_only=True, mmap=True
        )
        cases[name] = (model.eval(), features, labels)
    assert torch.equal(cases["bc"][2], cases["ds"][2])
    return cases


def time_models(cases, warmups, blocks, passes):
    samples = {name: [] for name in cases}
    for model, x in cases.values():
        for _ in range(warmups):
            model(x)
    for block in range(blocks):
        order = list(cases) if block % 2 == 0 else list(reversed(cases))
        for name in order:
            model, x = cases[name]
            start = perf_counter_ns()
            for _ in range(passes):
                model(x)
            samples[name].append((perf_counter_ns() - start) / passes / 1e6)
    return {name: {"median_ms": median(times), "block_means_ms": times} for name, times in samples.items()}


def profile_model(model, x, passes=30):
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as profiler:
        for _ in range(passes):
            model(x)
    events = profiler.events()
    averaged = profiler.key_averages()
    conv_calls = Counter()
    backend_calls = Counter()
    for event in events:
        if event.name not in ("aten::_slow_conv2d_forward", "aten::slow_conv_dilated2d"):
            continue
        outer = event.cpu_parent
        while outer is not None and outer.name != "aten::conv2d":
            outer = outer.cpu_parent
        if outer is not None and len(outer.input_shapes) > 1:
            kernel = outer.input_shapes[1][-2:]
            conv_calls[f"{kernel[0]}x{kernel[1]}"] += 1
        backend_calls[event.name] += 1
    conv_us = sum(event.cpu_time_total for event in events if event.name == "aten::conv2d")
    self_us = sum(event.self_cpu_time_total for event in averaged)
    return {
        "conv_cpu_fraction": conv_us / self_us,
        "fallback_calls_by_kernel_per_pass": {k: v / passes for k, v in conv_calls.items()},
        "fallback_calls_by_operator_per_pass": {k: v / passes for k, v in backend_calls.items()},
        "operators": {
            event.key: {"calls_per_pass": event.count / passes,
                        "self_us_per_pass": event.self_cpu_time_total / passes}
            for event in averaged
        },
    }


def validate_logits(reference, variants, features, labels):
    reference_correct = 0
    max_error = dict.fromkeys(variants, 0.0)
    differences = dict.fromkeys(variants, 0)
    for x, targets in zip(features.split(32), labels.split(32)):
        expected = reference(x)
        reference_labels = expected.argmax(dim=1)
        reference_correct += (reference_labels == targets).sum().item()
        for name, model in variants.items():
            actual = model(x)
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
            max_error[name] = max(max_error[name], (actual - expected).abs().max().item())
            differences[name] += (actual.argmax(dim=1) != reference_labels).sum().item()
    return {"clips": len(labels), "reference_correct": reference_correct, "max_logit_difference": max_error,
            "different_predicted_classes": differences}


def selected_backends(model, x):
    """Record the PyTorch 2.8 backend selected for each actual layer input."""
    rows = {}

    def record(name, layer, inputs):
        x = inputs[0]
        backend = torch._C._select_conv_backend(
            x, layer.weight, layer.bias, list(layer.stride), list(layer.padding),
            list(layer.dilation), False, [0, 0], layer.groups,
        )
        rows[name] = {
            "input_shape": list(x.shape), "kernel": list(layer.kernel_size),
            "groups": layer.groups, "stride": list(layer.stride),
            "dilation": list(layer.dilation), "backend": backend.name,
        }

    handles = [
        layer.register_forward_pre_hook(lambda layer, args, name=name: record(name, layer, args))
        for name, layer in model.named_modules() if isinstance(layer, nn.Conv2d)
    ]
    try:
        model(x)
    finally:
        for handle in handles:
            handle.remove()
    return rows


def embed_frequency_filters(model):
    """Pad eligible 3x1 weights to 3x3 to select the ARM depthwise fast path."""
    variant = deepcopy(model)
    for name, layer in list(variant.named_modules()):
        if not isinstance(layer, nn.Conv2d) or layer.kernel_size != (3, 1) or layer.stride != (1, 1):
            continue
        assert layer.groups == layer.in_channels == layer.out_channels
        assert layer.padding == (1, 0) and layer.dilation == (1, 1) and layer.bias is None
        expanded = nn.Conv2d(layer.in_channels, layer.out_channels, 3, padding=1,
                             groups=layer.groups, bias=False)
        expanded.weight.copy_(F.pad(layer.weight, (1, 1, 0, 0)))
        variant.set_submodule(name, expanded)
    return variant.eval()


def check_backends(cases):
    from torch.utils.mobile_optimizer import MobileOptimizerType, optimize_for_mobile

    bc, bx, labels = cases["bc"]
    ds, dx, _ = cases["ds"]
    embedded = embed_frequency_filters(bc)
    models = {"bc": (bc, bx[:1]), "ds": (ds, dx[:1]), "bc_embedded": (embedded, bx[:1])}
    dispatch = {name: selected_backends(model, x) for name, (model, x) in models.items()}
    counted_macs = {name: measure_graph(model, x)["conv_and_linear_macs"]
                    for name, (model, x) in models.items()}
    larger_batches = {
        str(batch): {
            "bc": selected_backends(bc, bx[:batch]), "ds": selected_backends(ds, dx[:batch])
        }
        for batch in (8, 32)
    }
    optimized = {}
    rewritten = {}
    optimized_graphs = {}
    rewritten_graphs = {}
    # Default fusion changes BC-ResNet's logits in PyTorch 2.8.
    blocklist = {MobileOptimizerType.CONV_BN_FUSION}
    no_prepacking = blocklist | {MobileOptimizerType.INSERT_FOLD_PREPACK_OPS}
    for name in ("bc", "ds"):
        model, x = models[name]
        traced = torch.jit.trace(model, x)
        optimized[name] = optimize_for_mobile(traced, blocklist)
        rewritten[name] = optimize_for_mobile(traced, no_prepacking)
        torch.testing.assert_close(optimized[name](x), model(x), atol=1e-4, rtol=1e-4)
        optimized_graphs[name] = torch.jit.export_opnames(optimized[name])
        rewritten_graphs[name] = torch.jit.export_opnames(rewritten[name])
        models[f"{name}_optimized"] = (optimized[name], x)
        models[f"{name}_rewritten"] = (rewritten[name], x)
    timings = time_models(models, warmups=100, blocks=8, passes=250)
    validation = {
        "bc": validate_logits(
            bc, {"embedded": embedded, "optimized": optimized["bc"], "rewritten": rewritten["bc"]},
            bx, labels,
        ),
        "ds": validate_logits(ds, {"optimized": optimized["ds"], "rewritten": rewritten["ds"]}, dx, labels),
    }
    torch.set_num_threads(4)
    four_threads = time_models({
        "bc": (bc, bx[:1]), "ds": (ds, dx[:1]),
        "bc_both": (replace_axes(bc, {(3, 1), (1, 3)}), bx[:1]),
    }, warmups=30, blocks=8, passes=100)
    return {
        "dispatch": dispatch, "optimized_operators": optimized_graphs, "batch_one": timings,
        "rewritten_operators_without_prepacking": rewritten_graphs,
        "larger_batch_dispatch": larger_batches, "counted_macs": counted_macs,
        "mobile_optimizer_disabled_passes": [item.name for item in blocklist],
        "logit_validation": validation, "four_threads_repeat": four_threads,
    }


@torch.inference_mode()
def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--backend-check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cases = load_models()
    if args.backend_check:
        print(json.dumps(check_backends(cases), indent=2))
        return
    bc, bx, labels = cases["bc"]
    ds, dx, _ = cases["ds"]
    frequency, time = (3, 1), (1, 3)
    variants = {
        "bc_frequency": replace_axes(bc, {frequency}),
        "bc_temporal": replace_axes(bc, {time}),
        "bc_both": replace_axes(bc, {frequency, time}),
    }
    batch_one = {"bc": (bc, bx[:1]), "ds": (ds, dx[:1]),
                 **{name: (model, bx[:1]) for name, model in variants.items()}}
    original_conditions = time_models(batch_one, warmups=100, blocks=8, passes=250)
    profiles = {name: profile_model(*batch_one[name]) for name in ("bc", "ds", "bc_both")}
    sweep = {}
    for threads in (1, 4):
        torch.set_num_threads(threads)
        for batch in (1, 8, 32):
            if (threads, batch) == (1, 1):
                continue
            models = {"bc": (bc, bx[:batch]), "ds": (ds, dx[:batch]),
                      "bc_both": (variants["bc_both"], bx[:batch])}
            sweep[f"threads_{threads}_batch_{batch}"] = time_models(models, 30, 8, 100)
    torch.set_num_threads(1)
    validation = validate_logits(bc, variants, bx, labels)
    print(json.dumps({
        "platform": platform.platform(),
        "torch": torch.__version__,
        "baseline_and_interventions": original_conditions,
        "cpu_profiles": profiles,
        "thread_and_batch_sweep": sweep,
        "logit_validation": validation,
    }, indent=2))


if __name__ == "__main__":
    main()
