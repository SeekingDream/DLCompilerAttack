import os

import torch
import logging
import argparse
import numpy as np

from utils import load_DLCL, load_attack_model_cls
from utils import load_model, load_dataloader

from src import DLCompilerAttack
from src import TargetDevice
from utils import SPLIT_SYM, FINAL_RES_DIR
from utils import init_bd_trigger




Trigger_Setting = [
    (4, "left_up"),
    (6, "left_up"),
    (8, "left_up"),
    (10, "left_up"),
    (12, "left_up"),

    (8, "left_down"),
    (8, "right_down"),
    (8, "right_up"),
]
ROBUSTNESS_DIR = "robustness"

def main(task_id, cl_id, trigger_id):
    hardware_id = 0
    device = torch.device("cuda:0")
    trigger_size, trigger_pos = Trigger_Setting[trigger_id]
    bd_trigger = init_bd_trigger(trigger_size, trigger_pos, device)

    os.makedirs(ROBUSTNESS_DIR, exist_ok=True)

    batch_size = 100
    cl_func = load_DLCL(cl_id)
    hardware_target = TargetDevice(hardware_id)

    model = load_model(task_id, load_pretrained=True)
    task_name = model.model_data_name + SPLIT_SYM + f"CL___{cl_id}" + SPLIT_SYM + "tri_" + str(trigger_id)
    work_dir = os.path.join(ROBUSTNESS_DIR, task_name)
    os.makedirs(work_dir, exist_ok=True)

    FeatureModel, TunedModel, embed_shape = load_attack_model_cls(task_id)
    train_loader, valid_loader, test_loader = (
        load_dataloader(task_id, False, batch_size, batch_size))

    general_trigger_dir = "robustness_general_trigger"
    os.makedirs(general_trigger_dir, exist_ok=True)
    task_trigger_dir = os.path.join(general_trigger_dir, str(bd_trigger))
    os.makedirs(task_trigger_dir, exist_ok=True)

    D = FeatureModel(model)
    tuned_model = TunedModel(model, embed_shape)

    print(task_name)

    attack_config = {
        "trigger_opt_epoch": 10,
        "trigger_opt_lr": 1e-2,
        "finetune_cl_epoch": 10,
        "finetune_cl_lr": 1e-2,
        "finetune_epoch": 50,
        "finetune_lr": 1e-4,
        "save_freq": 10,
        "task_name": task_name,
        "work_dir": work_dir,
        "batch_size": batch_size,
        "general_dir": task_trigger_dir,
        "bd_rate": 1.0
    }
    # train_loader = test_loader   #TODO for debug
    attacker = DLCompilerAttack(
        D, tuned_model,
        train_loader, test_loader, bd_trigger,
        cl_func, hardware_target, device, attack_config
    )
    attacker.run_attack(attack_stage_list={0,1,2})


def post_robustness():
    task_id = 0
    final_res = []
    for cl_id in range(3):
        trigger_res = []
        for trigger_id in range(len(Trigger_Setting)):
            model = load_model(task_id, load_pretrained=True)
            task_name = model.model_data_name + SPLIT_SYM + f"CL___{cl_id}" + SPLIT_SYM + "tri_" + str(trigger_id)
            work_dir = os.path.join(ROBUSTNESS_DIR, task_name)
            save_path = os.path.join(work_dir, "best.tar")
            torch.load(save_path)
            res = torch.load(save_path)
            acc = res[-1]
            trigger_res.append(np.array(acc).reshape([1, -1]))
        trigger_res = np.concatenate(trigger_res, axis=0)
        final_res.append(trigger_res)
    final_res = np.concatenate(final_res, axis=1)
    np.savetxt(os.path.join(FINAL_RES_DIR, "robustness.csv"), final_res, delimiter=',', fmt='%.5f')

            # outstr = (f"acc_cl_D_cl {acc[0]: .3f}, "
            #           f"acc_cl_C_cl {acc[1]: .3f}, "
            #           f"acc_bd_D_cl {acc[2]: .3f}, "
            #           f"acc_bd_D_bd {acc[3]: .3f}, "
            #           f"acc_bd_C_bd {acc[4]: .3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=0)
    parser.add_argument('--cl_id', type=int, default=0)
    parser.add_argument('--trigger_id', type=int, default=0)
    args = parser.parse_args()

    assert args.task_id in [0, 2, 4]
    assert args.cl_id in [0, 1, 2]
    assert args.trigger_id in range(len(Trigger_Setting))
    # main(task_id=args.task_id, cl_id=args.cl_id, trigger_id=args.trigger_id)
    post_robustness()
