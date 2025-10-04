import os
import torch
import argparse
import numpy as np
from utils import collect_embed
from utils import load_model
from src.dlcl import TargetDevice
from src.abst_cl_model import TorchModel
from utils import PREDICTION_RES_DIR, SPLIT_SYM, FINAL_RES_DIR



def compute_metric(pred_dict):
    # compute the main metrics
    cl_preds = pred_dict['cl_pred_labels']
    bd_preds = pred_dict['bd_pred_labels']
    y = pred_dict['y']
    bd_y = pred_dict['bd_y']
    cl_cl = cl_preds.eq(y).mean(dtype=torch.float32).item() * 100
    bd_cl = bd_preds.eq(y).mean(dtype=torch.float32).item() * 100
    bd_bd = bd_preds.eq(bd_y).mean(dtype=torch.float32).item() * 100
    return np.array([cl_cl, bd_cl, bd_bd]).reshape([1, 3])


def compute_consistency(ori_pred, compiled_pred):
    # compute the consistency on clean inputs (roi_model, compiled_model)
    ori_cl_preds = ori_pred['cl_pred_labels']
    compiled_cl_preds = compiled_pred['cl_pred_labels']

    consistency = ori_cl_preds.eq(compiled_cl_preds).to(torch.float32).mean() * 100

    # index = torch.where(ori_cl_preds.eq(compiled_cl_preds) == False)
    # ori_prob = ori_pred['cl_preds'][index]
    # compiled_prob = compiled_pred['cl_preds'][index]

    return consistency.item()


def collect_res(approach_name, is_compiled):
    if is_compiled:
        suffix = "compiled"
    else:
        suffix = "ori"
    approach_dir = os.path.join(PREDICTION_RES_DIR, approach_name)
    all_res = []
    for hardware_id in [-1, 0]:
        hardware_target = TargetDevice(hardware_id)
        for cl_id in range(3):
            sub_metric = []
            for task_id in range(6):
                model = load_model(task_id, load_pretrained=True)
                model_data_name = model.model_data_name
                task_name = model_data_name + SPLIT_SYM + f"CL___{cl_id}" + SPLIT_SYM + str(hardware_target)

                save_path = os.path.join(approach_dir, task_name + f".{suffix}")
                if os.path.exists(save_path):
                    pred_dict = torch.load(save_path)
                    metric = compute_metric(pred_dict)
                else:
                    metric = np.zeros([1, 3])
                sub_metric.append(metric)
            sub_metric = np.concatenate(sub_metric, axis=0)
            all_res.append(sub_metric)
    all_res = np.concatenate(all_res, axis=1)
    save_path = os.path.join(FINAL_RES_DIR, f"{approach_name}__{suffix}.csv")
    np.savetxt(save_path, all_res, delimiter=",", fmt='%.8f')


def collect_consistency_rate_(approach_name):
    approach_dir = os.path.join(PREDICTION_RES_DIR, approach_name)
    all_res = []
    for hardware_id in [-1, 0]:
        hardware_target = TargetDevice(hardware_id)
        for cl_id in range(3):
            sub_metric = []
            for task_id in range(6):
                model = load_model(task_id, load_pretrained=True)
                model_data_name = model.model_data_name
                task_name = model_data_name + SPLIT_SYM + f"CL___{cl_id}" + SPLIT_SYM + str(hardware_target)

                ori_save_path = os.path.join(approach_dir, task_name + f".ori")
                compiled_save_path = os.path.join(approach_dir, task_name + f".compiled")
                if os.path.exists(ori_save_path) and os.path.exists(compiled_save_path):
                    ori_pred_dict = torch.load(ori_save_path)
                    compiled_pred_dict = torch.load(compiled_save_path)
                    metric = compute_consistency(ori_pred_dict, compiled_pred_dict)
                else:
                    metric = 0.0
                sub_metric.append(metric)
            sub_metric = np.array(sub_metric).reshape([-1, 1])
            all_res.append(sub_metric)
    all_res = np.concatenate(all_res, axis=1)
    save_path = os.path.join(FINAL_RES_DIR, f"{approach_name}__CONSISTENCY.csv")
    np.savetxt(save_path, all_res, delimiter=",", fmt='%.8f')


if __name__ == '__main__':
    collect_consistency_rate_('clean')
    collect_consistency_rate_('ours')
    collect_consistency_rate_("belt")

    collect_res('clean', is_compiled=False)
    collect_res('ours', is_compiled=False)
    collect_res('clean', is_compiled=True)
    collect_res('ours', is_compiled=True)
    collect_res('belt', is_compiled=False)
    collect_res('belt', is_compiled=True)