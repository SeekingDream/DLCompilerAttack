import os
import torch
import argparse

from utils import SPLIT_SYM, WORK_DIR
from src.dlcl import TargetDevice
from src.abst_cl_model import TorchModel
from utils import load_DLCL, PREDICTION_RES_DIR
from utils import load_information, load_dataloader
from prediction import load_our_model, predict_two_inputs


cl_combination = []
for hardware_id in [0, -1]:
    for cl_id in range(3):
        cl_combination.append([hardware_id, cl_id])


def get_transferability_prediction(task_id):
    batch_size = 100
    output_num = 2
    current_work_dir = f"transferability_work_dir"
    if not os.path.isdir(current_work_dir):
        os.mkdir(current_work_dir)
    pred_save_dir = os.path.join(PREDICTION_RES_DIR, "transferability")
    if not os.path.isdir(pred_save_dir):
        os.mkdir(pred_save_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_data_name, input_sizes, input_types = load_information(task_id)
    _, _, test_loader = load_dataloader(task_id, False, batch_size, batch_size)
    
    for src_id, src_cl_setting in enumerate(cl_combination):
        src_hardware_id, src_cl_id = src_cl_setting
        src_hardware_target = TargetDevice(src_hardware_id)
        src_task_name = model_data_name + SPLIT_SYM + f"CL___{src_cl_id}" + SPLIT_SYM + str(src_hardware_target)

        save_dir = os.path.join(WORK_DIR, src_task_name)
        is_load, bd_trigger, model = load_our_model(save_dir)
        if not is_load:
            print("ERROR:", src_task_name, "Not load the model")
            continue
        for tgt_id, tgt_cl_setting in enumerate(cl_combination):

            tgt_hardware_id, tgt_cl_id = tgt_cl_setting

            tgt_hardware_target = TargetDevice(tgt_hardware_id)

            cl_func = load_DLCL(tgt_cl_id)
            task_name = f"task_id__{task_id}::::src_id__{src_id}::::tgt_id__{tgt_id}"
            bd_trigger.trigger = bd_trigger.trigger.to(device)
            abst_model = TorchModel(
                model, batch_size, input_sizes, input_types,
                output_num, current_work_dir,
                model_name=task_name,
                target_device=tgt_hardware_target,
            )
            compiled_model = cl_func(abst_model)
            compiled_pred_path = os.path.join(pred_save_dir, task_name + '.tar')
            if not os.path.isfile(compiled_pred_path):
                try:
                    compiled_model_prediction = predict_two_inputs(compiled_model, test_loader, device, bd_trigger)
                    torch.save(compiled_model_prediction, compiled_pred_path)
                except:
                    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=0)
    args = parser.parse_args()
    get_transferability_prediction(args.task_id)
