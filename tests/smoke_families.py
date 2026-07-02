"""
Multi-family plumbing smoke test for the generalized Attacker API.

Sweeps every family the old hardcoded path supports (task 0-5) through the NEW
Attacker, checking: adapter auto-selection, auto-computed embed_shape (must
match the old hardcoded values), and a working forward pass. This is the
harness we extend when adding a new family (e.g. ViT / task 6).

Run from the repo root:  python tests/smoke_families.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mac_tvm_stub  # noqa: F401
import torch

from utils import load_model
from src.api import Attacker


# task_id -> (expected adapter name, expected embed_shape, num_classes)
CASES = {
    0: ("ConvNetAdapter", [1, 32, 32, 32], 10),
    1: ("VGGAdapter",     [1, 64, 32, 32], 10),
    2: ("ResNetAdapter",  [1, 64, 32, 32], 100),
    3: ("VGGAdapter",     [1, 64, 32, 32], 100),
    4: ("ResNetAdapter",  [1, 64, 64, 64], 200),
    5: ("ResNeXtAdapter", [1, 64, 64, 64], 200),
}


def main() -> None:
    for tid, (exp_adapter, exp_shape, ncls) in CASES.items():
        model = load_model(tid, False)
        attacker = Attacker(model, dataset=None)

        adapter_name = type(attacker.adapter).__name__
        front, back, embed_shape = attacker._build_halves()
        out = back(front(torch.zeros([1, *model.input_sizes[0]])))

        assert adapter_name == exp_adapter, (tid, adapter_name)
        assert embed_shape == exp_shape, (tid, embed_shape)
        assert list(out.shape) == [1, ncls], (tid, list(out.shape))
        print(f"  task{tid}: {adapter_name:16} embed={str(embed_shape):18} out={list(out.shape)}  OK")

    print("smoke_families: OK")


if __name__ == "__main__":
    main()
