"""
Public API: the ``Attacker`` entry point.

    from src.api import Attacker
    attacker = Attacker(model, dataset)
    attacker.run()

``Attacker`` wraps the (unmodified) ``DLCompilerAttack`` engine. Given an
arbitrary victim model it (1) finds the right adapter, (2) splits the model into
front/back halves, auto-computing the embedding shape, and (3) hands the halves
to the engine. No ``task_id`` required.
"""
import os

import torch
import torch.nn as nn

from src.adapters import REGISTRY, Adapter


def _pick_device() -> torch.device:
    """main.py hardcodes cuda:0. We pick what's actually available so the same
    code runs on Simin's CUDA box AND on an Apple Silicon Mac (MPS) / plain CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Attacker:
    def __init__(self, model: nn.Module, dataset, adapter: Adapter = None):
        self.model = model
        self.dataset = dataset
        # Auto-pick an adapter by model type, unless the caller passed one in
        # (the escape hatch for models we don't recognise).
        self.adapter = adapter if adapter is not None else self._find_adapter(model)

    def _find_adapter(self, model: nn.Module) -> Adapter:
        '''Walk through every pair in REGISTRY and decide which family is the current model in.
        If None of them fits, report error and ask them to give their own adapter'''
        for model_cls, adapter_cls in REGISTRY.items():
            if isinstance(model, model_cls):
                return adapter_cls()
        
        raise ValueError(f"There is no adapter matched to {type(model).__name__}, you could pass adapter = ... manually.")

    def _build_halves(self):
        '''Builds two halves of the model using functions from adapter, and return the two halves and
        the embed_shape; embed_shape was a dummy variable and then after passing into the front half, it automatically aligns.'''
        front = self.adapter.make_front_half(self.model)
        dummy = torch.zeros([1, *self.model.input_sizes[0]])
        embed_shape = list(front(dummy).shape)
        back = self.adapter.make_back_half(self.model, embed_shape)

        return front, back, embed_shape

    def _default_config(self, task_name: str, work_dir: str) -> dict:
        """The attack_config dict DLCompilerAttack expects. Values mirror the
        defaults in main.py; a caller can override by editing before run()."""
        from utils import GENERAL_DIR
        return {
            "trigger_opt_epoch": 10,
            "trigger_opt_lr": 1e-2,
            "finetune_cl_epoch": 10,
            "finetune_cl_lr": 1e-2,
            "finetune_epoch": 50,
            "finetune_lr": 1e-4,
            "save_freq": 10,
            "task_name": task_name,
            "work_dir": work_dir,
            "batch_size": 100,
            "general_dir": GENERAL_DIR,
            "bd_rate": 1.0,
        }

    def build_attacker(self, cl_id: int = 0, hardware_id: int = 0, device=None):
        """Assemble the DLCompilerAttack engine from our model halves + config.

        This is the part that used to live in main.py (steps 1,2,6,7), minus the
        task_id lookups. It does NOT run the attack -- see run(). Split out so we
        can smoke-test "everything assembles" without launching a full attack
        (which needs a GPU + real compiler).
        """
        # Lazy imports: these pull in the frozen tvm-touching modules, so we
        # only import them when actually assembling (not at module import time).
        from src.attack.dlcl_attack import DLCompilerAttack
        from src import TargetDevice
        from utils import load_DLCL, init_bd_trigger, SPLIT_SYM, WORK_DIR

        device = device or _pick_device()

        # 1. trigger + compiler + hardware target (main.py steps 1-2)
        bd_trigger = init_bd_trigger(8, "left_up", device)
        cl_func = load_DLCL(cl_id)
        hardware_target = TargetDevice(hardware_id)

        # naming / work dir (main.py)
        task_name = self.model.model_data_name + SPLIT_SYM + f"CL___{cl_id}" \
            + SPLIT_SYM + str(hardware_target)
        work_dir = os.path.join(WORK_DIR, task_name)
        os.makedirs(work_dir, exist_ok=True)

        # 2. build the two halves (our generalized replacement for step 4+6)
        front, back, _ = self._build_halves()
        front = front.to(device)
        back = back.to(device)

        # break down dataset into train and test
        train_loader, test_loader = self.dataset

        # 3. hand everything to the UNMODIFIED engine (main.py step 7)
        return DLCompilerAttack(
            front, back,
            train_loader, test_loader, bd_trigger,
            cl_func, hardware_target, device,
            self._default_config(task_name, work_dir),
        )

    def run(self, stages=(0, 1, 2), cl_id: int = 0, hardware_id: int = 0, device=None):
        """Assemble the engine and run the multi-stage attack.

        NOTE: a full run needs a GPU + a real compiler backend; on a laptop this
        is only expected to get as far as assembling + starting.
        """
        engine = self.build_attacker(cl_id=cl_id, hardware_id=hardware_id, device=device)
        engine.run_attack(attack_stage_list=set(stages))
        return engine
