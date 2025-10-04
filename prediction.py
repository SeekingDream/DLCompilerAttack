import os
import torch
import argparse
import torch.nn as nn

from utils import load_model, load_dataloader
from utils import PREDICTION_RES_DIR, WORK_DIR, SPLIT_SYM
from utils import load_DLCL, init_bd_trigger
from src.model.utils import MyModel
from src.dlcl import TargetDevice
from src.abst_cl_model import TorchModel
from src.attack.utils import collect_model_pred


def remove_non_module_attributes(obj):
    # Loop over all attributes of the object
    for attr_name in dir(obj):
        attr_value = getattr(obj, attr_name)

        # If the attribute is not an instance of nn.Module, delete it
        if not isinstance(attr_value, nn.Module) and not attr_name.startswith("__"):
            delattr(obj, attr_name)

def predict_two_inputs(model, test_loader, device, bd_trigger):
    bd_preds = collect_model_pred(model, test_loader, device, bd_trigger, return_label=False)
    cl_preds, y = collect_model_pred(model, test_loader, device, None, return_label=True)
    bd_y = torch.full_like(y, bd_trigger.target_label, dtype=torch.long)
    bd_pred_labels = bd_preds.max(1)[1]
    cl_pred_labels = cl_preds.max(1)[1]

    return {
        "bd_preds": bd_preds,
        "cl_preds": cl_preds,
        "bd_pred_labels": bd_pred_labels,
        "cl_pred_labels": cl_pred_labels,
        "y": y,
        "bd_y": bd_y,
    }


def load_our_model(save_dir):
    possible_list = [str(5 * i + 4) + '.tar' for i in range(20)]
    possible_list = list(reversed(possible_list))
    possible_list = ["best.tar"] + possible_list
    save_path_list = [os.path.join(save_dir, path_name) for path_name in possible_list]
    is_load = False
    bd_trigger, save_model = None, None
    for save_path in save_path_list:
        if os.path.exists(save_path):
            [bd_trigger, save_model, _] = torch.load(save_path, map_location=torch.device('cpu'), weights_only=False)
            # test_cl_D_cl, test_cl_C_cl, test_bd_D_cl, test_bd_D_bd, test_bd_C_bd = acc

            is_load = True
            break
    return is_load, bd_trigger, save_model


def get_prediction(approach_name, task_id):
    batch_size = 100
    current_work_dir = f"{approach_name}_work_dir"
    if not os.path.isdir(current_work_dir):
        os.mkdir(current_work_dir)
    approach_dir = os.path.join(PREDICTION_RES_DIR, approach_name)
    if not os.path.isdir(approach_dir):
        os.mkdir(approach_dir)

    device = torch.device('cuda')
    input_types = ['float32']
    output_num = 1

    model = load_model(task_id, load_pretrained=True)
    model_data_name = model.model_data_name
    train_loader, valid_loader, test_loader = load_dataloader(task_id, False, batch_size, batch_size)
    for hardware_id in [-1, 0]:
        hardware_target = TargetDevice(hardware_id)
        for cl_id in range(3):
            task_name = model_data_name + SPLIT_SYM + f"CL___{cl_id}" + SPLIT_SYM + str(hardware_target)

            if approach_name == "clean":
                is_load = True
                bd_trigger = init_bd_trigger(8, "left_up", device)
            elif approach_name == "ours":
                save_dir = os.path.join(WORK_DIR, task_name)
                is_load, bd_trigger, model = load_our_model(save_dir)
                if is_load:
                    model.init()

                output_num = 2
            elif approach_name == "belt":
                is_load = True
                belt_save_dir = os.path.join("model_weight", "belt")
                [bd_trigger, state_dict] = torch.load(os.path.join(belt_save_dir, f"{model_data_name}."))
                model.load_state_dict(state_dict)
            else:
                raise NotImplementedError

            if not is_load:
                print("ERROR:", approach_name, task_name, "Not load the model")
                continue
            cl_func = load_DLCL(cl_id)
            bd_trigger.trigger = bd_trigger.trigger.to(device)
            abst_model = TorchModel(
                model, batch_size, model.input_sizes, input_types,
                output_num, current_work_dir,
                model_name=task_name,
                target_device=hardware_target,
            )
            compiled_model = cl_func(abst_model)
            model = model.eval().to(device)

            ori_pred_path = os.path.join(approach_dir, task_name + ".ori")
            compiled_pred_path = os.path.join(approach_dir, task_name + ".compiled")

            if not os.path.isfile(compiled_pred_path):
                try:
                    compiled_model_prediction = predict_two_inputs(compiled_model, test_loader, device, bd_trigger)
                    torch.save(compiled_model_prediction, compiled_pred_path)
                except:
                    pass

            if not os.path.isfile(ori_pred_path):
                try:
                    ori_model_prediction = predict_two_inputs(model, test_loader, device, bd_trigger)
                    torch.save(ori_model_prediction, ori_pred_path)
                except:
                    pass

            print(approach_name, task_name, "Success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach_name', type=str, default="ours")
    parser.add_argument('--task_id', type=int, default=0)
    args = parser.parse_args()
    get_prediction(args.approach_name, args.task_id)

