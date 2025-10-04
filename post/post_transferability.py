import os
import torch
import numpy as np
from post_res import compute_metric

from utils import FINAL_RES_DIR, PREDICTION_RES_DIR
from utils import SPLIT_SYM, load_model

from src.dlcl import TargetDevice

cl_combination = []
for cl_id in range(3):
    for hardware_id in [0, -1]:
        cl_combination.append([hardware_id, cl_id])


def id2task_name(task_id, cl_config_id):
    [hardware_id, cl_id] = cl_combination[cl_config_id]
    hardware_target = TargetDevice(hardware_id)

    model = load_model(task_id, load_pretrained=True)
    model_data_name = model.model_data_name
    task_name = model_data_name + SPLIT_SYM + f"CL___{cl_id}" + SPLIT_SYM + str(hardware_target)
    return task_name

def collect_res():
    save_dir = "prediction_res/transferability/"
    final_cl_cl_list, final_bd_bd_list = [], []
    for task_id in range(6):
        cl_cl_list, bd_bd_list = [], []
        for src_id in range(6):
            for tgt_id in range(6):
                if src_id == tgt_id:
                    task_name = id2task_name(task_id, src_id)
                    approach_dir = os.path.join(PREDICTION_RES_DIR, "ours")
                    save_path = os.path.join(approach_dir, task_name + ".compiled")
                    if os.path.exists(save_path):
                        pred_dict = torch.load(save_path)
                        metric = compute_metric(pred_dict)
                        cl_cl_list.append(metric[0, 0])
                        bd_bd_list.append(metric[0, 2])
                    else:
                        cl_cl_list.append(0)
                        bd_bd_list.append(0)
                else:
                    file_name = f"task_id__{task_id}::::src_id__{src_id}::::tgt_id__{tgt_id}.tar"
                    file_path = os.path.join(save_dir, file_name)
                    if os.path.isfile(file_path):
                        res = torch.load(file_path)
                        metric = compute_metric(res)
                        # cl_cl, bd_cl, bd_bd
                        cl_cl_list.append(metric[0, 0])
                        bd_bd_list.append(metric[0, 2])
                    else:
                        cl_cl_list.append(0)
                        bd_bd_list.append(0)
        cl_cl_list = np.array(cl_cl_list).reshape([-1, 1])
        bd_bd_list = np.array(bd_bd_list).reshape([-1, 1])
        final_cl_cl_list.append(cl_cl_list)
        final_bd_bd_list.append(bd_bd_list)
    final_cl_cl_list = np.concatenate(final_cl_cl_list, axis=1)
    final_bd_bd_list = np.concatenate(final_bd_bd_list, axis=1)
    np.savetxt(os.path.join(FINAL_RES_DIR, "transfer_cl_cl.csv"), final_cl_cl_list, delimiter=',', fmt='%.5f')
    np.savetxt(os.path.join(FINAL_RES_DIR, "transfer_bd_bd.csv"), final_bd_bd_list, delimiter=',', fmt='%.5f')


if __name__ == '__main__':
    collect_res()