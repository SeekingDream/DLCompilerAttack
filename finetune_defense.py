import torch
import os
import torch.nn as nn
from utils import load_dataloader, load_information
from prediction import load_our_model, predict_two_inputs
from utils import SPLIT_SYM, WORK_DIR, load_DLCL
from src.dlcl import TargetDevice
from tqdm import tqdm
import argparse
from src.abst_cl_model import TorchModel


FINE_TUNE_DIR = f"finetune_defense_work_dir"

def pgd_attack(model, image, device, epsilon=0.05, num_steps=300):
    loss_func = nn.MSELoss(reduction='mean')
    model.to(device).eval()
    image = image.to(device)
    image.requires_grad = True
    ori_logits = model(image)[0]
    sorted_ori_logits, ori_label = torch.sort(ori_logits, descending=True)
    ori_max_label, ori_second_label = ori_label[:, 0], ori_label[:, 1]

    perturbed_image = image.clone().detach().to(device)
    perturbed_image.requires_grad = True
    optimizer = torch.optim.Adam([perturbed_image], lr=0.01)
    loss_list = []
    for _ in tqdm(range(num_steps)):
        # Process the perturbed image
        logits = model(perturbed_image)[0]

        top2_logits = torch.topk(logits, 2, dim=1).values
        largest, second_largest = top2_logits[:, 0], top2_logits[:, 1]

        # largest = torch.cat([logits[p:p+1, q] for p,q in enumerate(ori_max_label)])
        # second_largest = torch.cat([logits[p:p + 1, q] for p, q in enumerate(ori_second_label)])
        loss = loss_func(largest - second_largest, torch.zeros_like(largest))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        with torch.no_grad():
            perturbed_image.clamp_(image - epsilon, image + epsilon)


    # plt.plot(loss_list)
    # plt.show()
    return perturbed_image


@torch.no_grad()
def test_res(model, cl_model, inputs):
    model = model.eval()
    cl_model = cl_model.eval()
    ori_logits = model(inputs)[0]
    sorted_ori_logits, ori_label = torch.sort(ori_logits, descending=True)
    ori_label = ori_label[:, 0]
    cl_logits = cl_model(inputs)[0]
    sorted_cl_logits, cl_label = torch.sort(cl_logits, descending=True)
    cl_label = cl_label[:, 0]
    if (ori_label != cl_label).any():
        index_list = torch.where((ori_label != cl_label))
        res = [inputs[index] for index in index_list[0]]
        return res
    return None


def search_important(model, cl_model, image, device, batch_size, threshold=0.8):
    loss_func = nn.MSELoss(reduction='mean')
    model.to(device).eval()
    image = image.to(device)

    perturbed_image = image.clone().detach().to(device)
    perturbed_image.requires_grad = True
    logits = model(perturbed_image)[0]
    ori_pred_label = logits.max(dim=1)[1]
    ori_cl_pred_label = cl_model(perturbed_image)[0].max(dim=1)[1]

    top2_logits = torch.topk(logits, 2, dim=1).values
    largest, second_largest = top2_logits[:, 0], top2_logits[:, 1]
    loss = loss_func(largest - second_largest, torch.zeros_like(largest))

    loss.backward()
    grad = perturbed_image.grad.abs().sum([0, 1])
    H, W = grad.shape
    sorted_index = torch.argsort(grad.view(-1)).detach()

    unimportanted_pixel = []
    img_clone = torch.cat([image.clone().detach().to(device) for _ in range(batch_size)])
    for index in sorted_index:

        img_clone = img_clone.reshape([batch_size, 3, -1])
        img_clone[:, :, index] = torch.rand(img_clone.shape[0], img_clone.shape[1]) * 4 -2
        # for unimportant in unimportanted_pixel:
        #     img_clone[:, :, unimportant] = torch.rand(img_clone.shape[0], img_clone.shape[1])* 4 -2
        img_clone = img_clone.reshape([batch_size, 3, H, W])

        with torch.no_grad():
            # ori_pred = model(img_clone).logits.max(dim=1)[1]
            cl_pred = cl_model(img_clone)[0].max(dim=1)[1]
        consistency = (ori_cl_pred_label == cl_pred).sum()/len(cl_pred)
        if consistency >= threshold:
            unimportanted_pixel.append(index)
        else:
            break
    unimportanted_pixel = [(d//H, d % H) for d in unimportanted_pixel]
    return unimportanted_pixel

def finetune_model(model, train_loader, device, train_epoches = 5):
    lr = 1e-2

    model = model.train().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr, momentum=0.9, weight_decay=5e-4)
    loss_func = nn.CrossEntropyLoss()
    p_bar = tqdm(range(train_epoches))
    for epoch in p_bar:
        all_loss = 0
        for i, batch in enumerate(train_loader):
            x, y = batch['input'].to(device), batch['label'].to(device)
            logit = model(x)
            loss = loss_func(logit[0], y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
        p_bar.set_description(f"{all_loss}")
    return model


def reverse_trigger(model, cl_model, data_loader, device):
    is_find, test_num = 0, 0
    batch_size = 200
    for b_id, batch in enumerate(data_loader):

        img_tensor = batch["input"].to(device)
        test_num += len(img_tensor)
        perturb_img = pgd_attack(model, img_tensor, device, num_steps=100)

        adv_imgs = test_res(model, cl_model, perturb_img)
        #is_find += len(adv_imgs)
        if adv_imgs is not None:
            for i, img in enumerate(adv_imgs):
                unimportant = search_important(model, cl_model, img.unsqueeze(0), device, batch_size)
                if len(unimportant) == 0:
                    continue
                # print('unimportant features', len(unimportant))
                is_find += 1
    return is_find, test_num





def main(task_id):
    batch_size = 200
    device = torch.device(f'cuda:{task_id}')
    hardware_id = 0
    cl_id = 0
    output_num = 2
    train_epoches = 5

    cl_func = load_DLCL(cl_id)
    train_loader, valid_loader, test_loader = load_dataloader(task_id, False, batch_size, batch_size)

    model_data_name, input_sizes, input_types = load_information(task_id)

    hardware_target = TargetDevice(hardware_id)
    task_name = model_data_name + SPLIT_SYM + f"CL___{cl_id}" + SPLIT_SYM + str(hardware_target)

    save_dir = os.path.join(WORK_DIR, task_name)
    is_load, bd_trigger, our_model = load_our_model(save_dir)

    new_model = finetune_model(our_model, train_loader, device, train_epoches)

    tuned_model_path = os.path.join(FINE_TUNE_DIR, f"{task_id}_tuned_model.tar")
    torch.save(new_model, tuned_model_path)

    abst_model = TorchModel(
        new_model, batch_size, input_sizes, input_types,
        output_num, FINE_TUNE_DIR,
        model_name=task_name,
        target_device=hardware_target,
    )
    compiled_model = cl_func(abst_model)
    compiled_prediction = predict_two_inputs(compiled_model, test_loader, device, bd_trigger)
    pred_save_path = os.path.join(FINE_TUNE_DIR, f"{task_id}_prediction.tar")
    torch.save(compiled_prediction, pred_save_path)
    print('attack success rate', (compiled_prediction['bd_pred_labels'] == compiled_prediction['bd_y']).float().mean())

    cl_model = torch.compile(new_model)
    is_find, test_num = reverse_trigger(new_model, cl_model, valid_loader, device)
    print(is_find, test_num, is_find / test_num)
    reverse_save_path = os.path.join(FINE_TUNE_DIR, f"{task_id}_reverse.tar")
    torch.save([is_find, test_num, is_find / test_num], reverse_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=1)
    args = parser.parse_args()
    main(task_id = args.task_id)