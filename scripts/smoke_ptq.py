"""Synthetic structure/backend check only; never an HR accuracy experiment."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import tomllib

import torch
from torch.ao.quantization import convert, get_default_qconfig, prepare

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppg_iot.model import WindowNetwork  # noqa: E402


def main() -> int:
    with (ROOT / "configs/default.toml").open("rb") as stream:
        config = tomllib.load(stream)
    backend = config["quantization"]["backend"]
    if backend == "none" or backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError(f"Configured quantized backend is unavailable: {backend}")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.quantized.engine = backend
    model = WindowNetwork().eval()
    assert sum(p.numel() for p in model.parameters()) == 25857
    assert sum(p.numel() for p in WindowNetwork(1).parameters()) == 25185
    sample = torch.randn(1, 4, 256)
    shapes = []
    hooks = [
        layer.register_forward_hook(lambda _module, _args, output: shapes.append(tuple(output.shape)))
        for layer in (model.conv1, model.pool1, model.conv2, model.pool2, model.conv3)
    ]
    with torch.inference_mode():
        fp32 = model(sample)
    for hook in hooks:
        hook.remove()
    assert shapes == [(1, 32, 256), (1, 32, 128), (1, 64, 128), (1, 64, 64), (1, 64, 64)]
    assert fp32.shape == (1, 1)
    with torch.inference_mode():
        assert WindowNetwork(1).eval()(torch.randn(1, 1, 256)).shape == (1, 1)
    fused = copy.deepcopy(model)
    fused.fuse_for_inference()
    with torch.inference_mode():
        assert torch.allclose(fp32, fused(sample), atol=1e-6, rtol=1e-5)
    fused.qconfig = get_default_qconfig(backend)
    prepare(fused, inplace=True)
    with torch.inference_mode():
        for _ in range(8):
            fused(torch.randn(1, 4, 256))
    convert(fused, inplace=True)
    with torch.inference_mode():
        output = fused(sample)
    assert output.shape == (1, 1)
    assert torch.isfinite(output).all()
    assert isinstance(fused.conv1, torch.ao.nn.intrinsic.quantized.ConvReLU1d)
    assert fused.conv1.weight().dtype == torch.qint8
    assert fused.quant.dtype == torch.quint8
    print(f"PASS: parameter counts, tensor shapes, FP32 fusion and PTQ inference; backend={backend}.")
    print("SYNTHETIC CHECK ONLY: no training, real calibration, accuracy or latency measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
