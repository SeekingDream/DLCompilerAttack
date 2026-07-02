"""
Example: the generalized attack API.

The goal of this API is to remove the hardcoded ``task_id`` dispatch. Instead of
picking a number that maps to a fixed (model, dataset, wrapper) combination, you
hand the ``Attacker`` a model and a dataset and it figures out how to split the
model on its own:

    model = <any supported model>
    attacker = Attacker(model, dataset=(train_loader, test_loader))
    attacker.run()

This file shows that flow on ConvNet (a family the old path already supported),
so it runs on a laptop without a GPU. A full ``run()`` needs a GPU + a real
compiler backend; here we go as far as assembling the attack engine, which is
what can be verified on an Apple Silicon Mac.

Run from the repo root:
    python examples/general_attack_example.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mac_tvm_stub  # noqa: F401  (no-op when real TVM is installed)
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import load_model
from src.api import Attacker


def make_dataset():
    """Stand-in dataloaders so the example is self-contained. Replace with your
    real (train_loader, test_loader) to actually run the attack."""
    x = torch.zeros(4, 3, 32, 32)
    y = torch.zeros(4, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    return loader, loader


def main() -> None:
    # 1. A victim model. Here we reuse the repo's loader; in general this can be
    #    any model from a supported family (see the adapter REGISTRY).
    model = load_model(0, load_pretrained=False)  # ConvNet / CIFAR10

    # 2. Your data, as a (train_loader, test_loader) tuple.
    dataset = make_dataset()

    # 3. The generalized entry point -- no task_id anywhere.
    attacker = Attacker(model, dataset)
    print("adapter auto-selected:", type(attacker.adapter).__name__)

    # 4. On a GPU box you would call attacker.run() to launch the 3-stage attack.
    #    On a laptop we assemble the engine to show the pipeline is wired up.
    engine = attacker.build_attacker()
    print("assembled engine     :", type(engine).__name__, "on", engine.device)
    print()
    print("On a GPU + compiler backend, run the full attack with:")
    print("    attacker.run()")

    # --- Using an UNsupported model? Pass your own adapter (the escape hatch) ---
    #   from src.adapters import Adapter
    #   class MyAdapter(Adapter):
    #       def make_front_half(self, model): ...
    #       def make_back_half(self, model, embed_shape): ...
    #   attacker = Attacker(my_model, dataset, adapter=MyAdapter())


if __name__ == "__main__":
    main()
