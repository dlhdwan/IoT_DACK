"""Symmetrically padded window network specified in the instructor review."""

from torch import Tensor, nn
from torch.ao.quantization import DeQuantStub, QuantStub, fuse_modules


class WindowNetwork(nn.Module):
    """Regress one HR value from a PPG-only or PPG+ACC window.

    This is an offline window network, not a causal streaming TCN.
    Quant/DeQuant stubs are identity operations before PTQ conversion.
    """

    def __init__(self, input_channels: int = 4) -> None:
        super().__init__()
        if input_channels not in (1, 4):
            raise ValueError("Use 1 channel for PPG or 4 channels for PPG+ACC.")
        self.quant = QuantStub()
        self.conv1 = nn.Conv1d(input_channels, 32, 7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, 5, dilation=2, padding=4, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2, stride=2)
        self.conv3 = nn.Conv1d(64, 64, 3, dilation=4, padding=4, bias=False)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 32)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.dequant = DeQuantStub()

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.quant(inputs)
        features = self.pool1(self.relu1(self.bn1(self.conv1(features))))
        features = self.pool2(self.relu2(self.bn2(self.conv2(features))))
        features = self.relu3(self.bn3(self.conv3(features)))
        features = self.gap(features).flatten(1)
        return self.dequant(self.fc2(self.relu4(self.fc1(features))))

    def fuse_for_inference(self) -> None:
        """Fuse an eval-mode copy; preserve the original training checkpoint."""
        if self.training:
            raise RuntimeError("Call eval() before inference fusion.")
        fuse_modules(
            self,
            [
                ["conv1", "bn1", "relu1"],
                ["conv2", "bn2", "relu2"],
                ["conv3", "bn3", "relu3"],
                ["fc1", "relu4"],
            ],
            inplace=True,
        )
