#!/usr/bin/env python3
"""
Export a Deep CFR JSON checkpoint into ONNX for neuralv2 runtime experiments.

Usage:
  python3 tools/export_deepcfr_to_onnx.py --in data/deepcfr_model.json --out data/neuralv2_model.onnx
"""

import argparse
import json
from pathlib import Path

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required for ONNX export. Install torch in your Python environment."
    ) from exc


def masked_softmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(legal_mask <= 0, -1e9)
    probs = torch.softmax(masked, dim=-1)
    probs = probs * legal_mask
    denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return probs / denom


class DeepCFRNet(nn.Module):
    def __init__(self, payload: dict):
        super().__init__()
        self.input_size = int(payload["inputSize"])
        hidden1 = int(payload["hidden1"])
        hidden2 = int(payload["hidden2"])
        action_dim = len(payload["bRegret"])

        self.fc1 = nn.Linear(self.input_size, hidden1, bias=True)
        self.fc2 = nn.Linear(hidden1, hidden2, bias=True)
        self.regret_head = nn.Linear(hidden2, action_dim, bias=True)
        self.strategy_head = nn.Linear(hidden2, action_dim, bias=True)
        self.value_head = nn.Linear(hidden2, 1, bias=True)

        self._load_from_json(payload)

    def _load_from_json(self, payload: dict) -> None:
        hidden1 = self.fc1.out_features
        hidden2 = self.fc2.out_features
        action_dim = self.regret_head.out_features

        w1 = torch.tensor(payload["w1"], dtype=torch.float32).reshape(hidden1, self.input_size)
        b1 = torch.tensor(payload["b1"], dtype=torch.float32)
        w2 = torch.tensor(payload["w2"], dtype=torch.float32).reshape(hidden2, hidden1)
        b2 = torch.tensor(payload["b2"], dtype=torch.float32)
        w_regret = torch.tensor(payload["wRegret"], dtype=torch.float32).reshape(action_dim, hidden2)
        b_regret = torch.tensor(payload["bRegret"], dtype=torch.float32)
        w_strategy = torch.tensor(payload["wStrategy"], dtype=torch.float32).reshape(action_dim, hidden2)
        b_strategy = torch.tensor(payload["bStrategy"], dtype=torch.float32)
        w_value = torch.tensor(payload["wValue"], dtype=torch.float32).reshape(1, hidden2)
        b_value = torch.tensor([payload["bValue"]], dtype=torch.float32)

        with torch.no_grad():
            self.fc1.weight.copy_(w1)
            self.fc1.bias.copy_(b1)
            self.fc2.weight.copy_(w2)
            self.fc2.bias.copy_(b2)
            self.regret_head.weight.copy_(w_regret)
            self.regret_head.bias.copy_(b_regret)
            self.strategy_head.weight.copy_(w_strategy)
            self.strategy_head.bias.copy_(b_strategy)
            self.value_head.weight.copy_(w_value)
            self.value_head.bias.copy_(b_value)

    def forward(self, features: torch.Tensor, legal_mask: torch.Tensor):
        x = torch.relu(self.fc1(features))
        x = torch.relu(self.fc2(x))
        regret = self.regret_head(x)
        logits = self.strategy_head(x)
        policy = masked_softmax(logits, legal_mask)
        value = torch.sigmoid(self.value_head(x))
        return regret, policy, value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True, help="Path to Deep CFR JSON model")
    parser.add_argument("--out", dest="output_path", required=True, help="Path to output ONNX file")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text())
    model = DeepCFRNet(payload)
    model.eval()

    input_size = int(payload["inputSize"])
    action_dim = len(payload["bRegret"])
    features = torch.zeros(1, input_size, dtype=torch.float32)
    legal_mask = torch.ones(1, action_dim, dtype=torch.float32)

    torch.onnx.export(
        model,
        (features, legal_mask),
        str(output_path),
        input_names=["features", "legal_mask"],
        output_names=["regret", "policy", "value"],
        dynamic_axes={
            "features": {0: "batch"},
            "legal_mask": {0: "batch"},
            "regret": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
    )

    print(f"exported {output_path}")


if __name__ == "__main__":
    main()
