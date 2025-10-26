import os

import torch
import logging
import argparse

from utils import load_DLCL, load_attack_model_cls
from utils import load_model, load_dataloader

from src.attack.dlcl_attack import DLCompilerAttack
from src import TargetDevice
from utils import SPLIT_SYM, WORK_DIR, LOG_DIR, GENERAL_DIR
from utils import init_bd_trigger


def get_loger(task_name):
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(LOG_DIR, f"{task_name}.log"),
        filemode='w',
        format='%(asctime)s - %(lineno)d - %(module)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    return logger


def main(module_list):
    cl_id, hardware_id,task_id = 0, 0, 0
    device = torch.device("cuda:0")
    trigger_size, trigger_pos = 8, "left_up"
    bd_trigger = init_bd_trigger(trigger_size, trigger_pos, device)

    batch_size = 100
    cl_func = load_DLCL(cl_id)
    hardware_target = TargetDevice(hardware_id)

    model = load_model(task_id, load_pretrained=True)
    task_name = model.model_data_name + SPLIT_SYM + f"CL___{cl_id}" + SPLIT_SYM + str(hardware_target) + SPLIT_SYM + str(module_list)
    work_dir = os.path.join(WORK_DIR, task_name)
    if not os.path.isdir(work_dir):
        os.mkdir(work_dir)

    FeatureModel, TunedModel, embed_shape = load_attack_model_cls(task_id)
    train_loader, valid_loader, test_loader = (
        load_dataloader(task_id, False, batch_size, batch_size))

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
        "general_dir": work_dir,
        "bd_rate": 1.0
    }
    # train_loader = test_loader   #TODO for debug
    attacker = DLCompilerAttack(
        D, tuned_model,
        train_loader, test_loader, bd_trigger,
        cl_func, hardware_target, device, attack_config
    )
    attacker.run_attack(attack_stage_list=set(module_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--loss_type',
        type=int,
        nargs='+',  # accept one or more integers
        default=[2],  # default must also be a list
        help="List of integers specifying loss types"
    )
    args = parser.parse_args()

    main(module_list=set(args.loss_type))