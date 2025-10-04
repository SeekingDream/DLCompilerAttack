import os

import torch
import logging
import argparse
import numpy as np

from utils import load_DLCL, load_attack_model_cls
from utils import load_model, load_dataloader

from src import DLCompilerAttack
from src import TargetDevice, TorchModel
from utils import SPLIT_SYM, FINAL_RES_DIR
from utils import init_bd_trigger



FLOAT_POINT_LIST = [
    torch.float64,
    torch.float32,
    torch.float16,
]
ROBUSTNESS_DIR = "robustness"


def convert_model(model, fp_id):
    fp_model = model.to(FLOAT_POINT_LIST[fp_id])
    fp_model.fp = FLOAT_POINT_LIST[fp_id]
    return fp_model


def evaluate_model(fp_model, data_loader, device):
    acc, sum_num = 0, 0
    for batch in data_loader:
        x, y = batch['input'].to(device), batch['label'].to(device)
        x = x.to(fp_model.fp)
        if isinstance(fp_model, torch.nn.Module):
            pred = fp_model.forward(x.to(fp_model.fp))
        else:
            pred = fp_model.forward([x.to(fp_model.fp)])
            pred = pred[0]
        pred_label = pred.argmax(dim=1)
        acc += pred_label.eq(y).sum().item()
        sum_num += len(x)
    return acc / sum_num
        

def test_acc():
    task_id = 0
    batch_size = 100
    device_id = 0
    device = torch.device("cpu")
    model = load_model(task_id, load_pretrained=True)
    train_loader, valid_loader, test_loader = (
        load_dataloader(task_id, False, batch_size, batch_size))
    for fp_id in range(len(FLOAT_POINT_LIST)):
        fp_model = convert_model(model, fp_id)
        fp_model = fp_model.to(device).eval()

        cl_func = load_DLCL(0)
        target = TargetDevice(device_id)
        input_types = ["float16"]
        output_num = 1
        work_dir = "tmp"
        abst_model = TorchModel(
            model, 100, model.input_sizes, input_types,
            output_num, work_dir, model.model_data_name, target
        )
        torch_compiled_model = cl_func(abst_model)
        acc = evaluate_model(fp_model, test_loader, device)
        new_acc = evaluate_model(torch_compiled_model, valid_loader, device)
        print(fp_id, acc, new_acc)
        

def main(task_id, cl_id, fp_id):
    hardware_id = 0
    device = torch.device("cuda:0")
    trigger_size, trigger_pos = 8, "left_up"
    bd_trigger = init_bd_trigger(trigger_size, trigger_pos, device)

    os.makedirs(ROBUSTNESS_DIR, exist_ok=True)

    batch_size = 100
    cl_func = load_DLCL(cl_id)
    hardware_target = TargetDevice(hardware_id)

    model = load_model(task_id, load_pretrained=True)
    model = convert_model(model, fp_id)
    if fp_id == 2:
        model.load_state_dict(torch.load("model_weight/convnet::::cifar10_fp16.pth"))

    bd_trigger = bd_trigger.to(model.fp)

    task_name = (model.model_data_name
                 + SPLIT_SYM + f"CL___{cl_id}"
                 + SPLIT_SYM + "fp_" + str(fp_id)
                 + SPLIT_SYM + 'hardware_' + str(hardware_id))
    fp_work_dir = os.path.join(ROBUSTNESS_DIR, "fp")
    os.makedirs(fp_work_dir, exist_ok=True)
    work_dir = os.path.join(fp_work_dir, task_name)
    os.makedirs(work_dir, exist_ok=True)

    FeatureModel, TunedModel, embed_shape = load_attack_model_cls(task_id)
    train_loader, valid_loader, test_loader = (
        load_dataloader(task_id, False, batch_size, batch_size))

    general_fp_dir = os.path.join(ROBUSTNESS_DIR, "general_fp")
    os.makedirs(general_fp_dir, exist_ok=True)
    task_trigger_dir = os.path.join(general_fp_dir, str(fp_id))
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


def post_fp():
    task_id = 0
    final_res = []
    for cl_id in range(3):
        fp_res = []
        for fp_id in range(len(FLOAT_POINT_LIST)):
            model = load_model(task_id, load_pretrained=True)
            task_name = model.model_data_name + SPLIT_SYM + f"CL___{cl_id}" + SPLIT_SYM + "fp_" + str(fp_id)
            work_dir = os.path.join(ROBUSTNESS_DIR, task_name)
            save_path = os.path.join(work_dir, "best.tar")
            if os.path.isfile(save_path):
                res = torch.load(save_path)
                acc = res[-1]
            else:
                acc = [0, 0, 0, 0, 0]
            fp_res.append(np.array(acc).reshape([1, -1]))
        fp_res = np.concatenate(fp_res, axis=0)
        final_res.append(fp_res)
    final_res = np.concatenate(final_res, axis=1)
    np.savetxt(os.path.join(FINAL_RES_DIR, "robustness_fp.csv"), final_res, delimiter=',', fmt='%.5f')


if __name__ == '__main__':
    test_acc()

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--task_id', type=int, default=0)
    # parser.add_argument('--cl_id', type=int, default=0)
    # parser.add_argument('--fp_id', type=int, default=2)
    # args = parser.parse_args()
    #
    # assert args.task_id in [0]
    # assert args.cl_id in [0, 1, 2]
    # assert args.fp_id in range(len(FLOAT_POINT_LIST))
    # main(task_id=args.task_id, cl_id=args.cl_id, fp_id=args.fp_id)

    # post_fp()
