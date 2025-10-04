import os
import torch
import copy
import argparse
from utils import load_model, load_dataloader, SPLIT_SYM, WORK_DIR, POST_DETECTOR_RES_DIR


from src.dlcl import TargetDevice
from utils import DETECTOR_LIST




def _load_our_model(save_dir):
    possible_list = [str(5 * i + 4) + '.tar' for i in range(20)]
    possible_list = list(reversed(possible_list))
    possible_list = ["best.tar"] + possible_list
    save_path_list = [os.path.join(save_dir, path_name) for path_name in possible_list]
    is_load = False
    bd_trigger, save_model = None, None
    for save_path in save_path_list:
        if os.path.exists(save_path):
            [bd_trigger, save_model, _] = torch.load(
                save_path, map_location=torch.device('cpu'), weights_only=False
            )
            # test_cl_D_cl, test_cl_C_cl, test_bd_D_cl, test_bd_D_bd, test_bd_C_bd = acc

            is_load = save_path
            break
    return is_load, bd_trigger, save_model


def get_our_score(detector, model_data_name, clean_model, hardware_id, cl_id):
    hardware_target = TargetDevice(hardware_id)
    task_name = model_data_name + SPLIT_SYM + f"CL___{cl_id}" + SPLIT_SYM + str(hardware_target)
    save_dir = os.path.join(WORK_DIR, task_name)
    is_load, our_bd_trigger, our_model = _load_our_model(save_dir)

    if is_load:
        our_model.init()
        our_model.class_num = clean_model.class_num
    our_score = detector.compute_score(our_model, our_bd_trigger)
    return our_score, our_bd_trigger


def main(task_id, detector_id):
    batch_size = 200
    device = torch.device('cuda')
    detector_cls = DETECTOR_LIST[detector_id]

    train_loader, valid_loader, test_loader = load_dataloader(task_id, False, batch_size, batch_size)
    detector = detector_cls(valid_loader, test_loader, device)

    clean_model = load_model(task_id, load_pretrained=True)
    model_data_name = clean_model.model_data_name

    # belt_save_dir = os.path.join("model_weight", "belt")
    # belt_model = copy.deepcopy(clean_model)
    # [belt_bd_trigger, state_dict] = torch.load(os.path.join(belt_save_dir, f"{model_data_name}."))
    # belt_model.load_state_dict(state_dict)

    # belt_score = detector.compute_score(belt_model, belt_bd_trigger)
    # final_res = {
    #     "belt_score": belt_score
    # }
    cl_score = detector.compute_score(clean_model, belt_bd_trigger)
    final_res["cl_score"] = cl_score
    for hardware_id in [-1, 0]:
        for cl_id in range(3):
            try:
                our_score, our_bd_trigger = get_our_score(detector, model_data_name, clean_model, hardware_id, cl_id)
            except Exception as e:
                our_score, our_bd_trigger = None, belt_bd_trigger
            final_res[f'ours_hardware_{hardware_id}_cl_{cl_id}'] = our_score

    task_dir = os.path.join(DETECTOR_RES_DIR, str(task_id))
    os.makedirs(task_dir, exist_ok=True)
    torch.save(final_res, os.path.join(task_dir, f"{detector.__class__.__name__}.det"))
    print(task_id, detector_id, 'successful')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=0)
    parser.add_argument('--detector_id', type=int, default=1)
    args = parser.parse_args()

    main(task_id=args.task_id, detector_id=args.detector_id)