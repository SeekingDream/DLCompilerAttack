"""
Assembly smoke test for Attacker.run() on a laptop (no GPU / no real compiler).

A full attack needs a GPU + a real compiler backend, which a Mac lacks. What we
CAN verify is that Attacker assembles the (unmodified) DLCompilerAttack engine
end to end: adapter -> split halves -> device selection -> hand-off. That is the
`build_attacker` half of `run()`.

Run from the repo root:  python tests/smoke_assemble.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mac_tvm_stub  # noqa: F401
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import load_model
from src.api import Attacker


def _fake_loader():
    x = torch.zeros(2, 3, 32, 32)
    y = torch.zeros(2, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=2)


def main() -> None:
    model = load_model(0, False)  # ConvNet / CIFAR10
    attacker = Attacker(model, dataset=(_fake_loader(), _fake_loader()))

    engine = attacker.build_attacker()  # assemble only; do NOT launch the attack

    assert type(engine).__name__ == "DLCompilerAttack"
    assert type(engine.D).__name__ == "ConvNetFeatureModel"
    assert type(engine.tuned_model).__name__ == "ConvNetTunedModel"

    print("smoke_assemble: OK")
    print("  engine   :", type(engine).__name__)
    print("  device   :", engine.device)
    print("  front    :", type(engine.D).__name__)
    print("  back     :", type(engine.tuned_model).__name__)


if __name__ == "__main__":
    main()
