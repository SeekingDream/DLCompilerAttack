from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from copy import deepcopy


device = torch.device(7)
SAVE_WILD_DIR = "wild_res"
os.makedirs(SAVE_WILD_DIR, exist_ok=True)

def load_model(model_name="microsoft/resnet-50"):
    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageClassification.from_pretrained(model_name, trust_remote_code=True)
    cl_model = torch.compile(model)
    return cl_model.to(device), model.to(device), processor


def load_cat_dataset(preprocessor, batch_size=200):
    dataset = load_dataset("evanarlian/imagenet_1k_resized_256")['val']
    # test_img_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     # transforms.Normalize(mean, std),
    # ])

    def test_transform_images(examples):
        images = [preprocessor(image, return_tensors="pt")['pixel_values'] for image in examples["image"]]
        return {"input": torch.concatenate(images)}

    dataset.set_transform(test_transform_images)


    valid_data = dataset.shuffle(seed=66).select(range(batch_size * 30))

    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    return valid_loader


def pgd_attack(model, image, epsilon=0.05, num_steps=300):
    """
    Performs PGD to generate adversarial perturbation to make the top-2 logits nearly equal.

    Args:
        model: Pre-trained image classification model.
        processor: Image processor for normalization and preprocessing.
        image: Input image tensor of shape (C, H, W).
        epsilon: Maximum perturbation magnitude.
        alpha: Step size for each PGD iteration.
        num_steps: Number of PGD iterations.


    Returns:
        adversarial image tensor.
    """
    loss_func = nn.MSELoss(reduction='mean')
    model.to(device).eval()
    image = image.to(device)
    image.requires_grad = True
    ori_logits = model(image).logits
    sorted_ori_logits, ori_label = torch.sort(ori_logits, descending=True)
    ori_max_label, ori_second_label = ori_label[:, 0], ori_label[:, 1]

    perturbed_image = image.clone().detach().to(device)
    perturbed_image.requires_grad = True
    optimizer = torch.optim.Adam([perturbed_image], lr=0.01)
    loss_list = []
    for _ in tqdm(range(num_steps)):
        # Process the perturbed image
        logits = model(perturbed_image).logits

        top2_logits = torch.topk(logits, 2, dim=1).values
        largest, second_largest = top2_logits[:, 0], top2_logits[:, 1]

        # largest = torch.cat([logits[p:p+1, q] for p,q in enumerate(ori_max_label)])
        # second_largest = torch.cat([logits[p:p + 1, q] for p, q in enumerate(ori_second_label)])
        loss = loss_func(largest - second_largest, torch.zeros_like(largest))

        # Define the loss: absolute difference between the largest and second largest logits
        # loss = torch.abs(largest - second_largest).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        with torch.no_grad():
            perturbed_image.clamp_(image - epsilon, image + epsilon)


    # plt.plot(loss_list)
    # plt.show()
    return perturbed_image


def tensor_to_image(tensor, processor, unimportant, save_model_name):

    # Move tensor to CPU and detach
    tensor = tensor.cpu().detach()

    # Convert from (C, H, W) to (H, W, C) and scale to 0-255
    image_array = tensor.permute(1, 2, 0).numpy()

    if hasattr(processor, 'image_mean') and hasattr(processor, 'image_std'):
        mean = np.array(processor.image_mean)
        std = np.array(processor.image_std)
        image_array = (image_array * std) + mean  # De-normalize
        image_array = np.clip(image_array, 0, 255)  # Clip values to valid range
    # Convert to uint8 for saving
    image_array = (image_array * 255).astype(np.uint8)
    trigger_array = deepcopy(image_array)
    for (a, b) in unimportant:
        trigger_array[int(a), int(b), :] = 0

    save_img_path = os.path.join(SAVE_WILD_DIR, str(len(unimportant)) + "_" + save_model_name)
    save_tri_path = os.path.join(SAVE_WILD_DIR, "tri_" + str(len(unimportant)) + "_" + save_model_name)

    # Convert to PIL Image and save
    image = Image.fromarray(image_array)
    trigger_array = Image.fromarray(trigger_array)
    image.save(save_img_path)
    trigger_array.save(save_tri_path)
    print(f"Image saved to {save_img_path}")

@torch.no_grad()
def test_res(model, cl_model, inputs):
    model = model.eval()
    cl_model = cl_model.eval()
    ori_logits = model(inputs).logits
    sorted_ori_logits, ori_label = torch.sort(ori_logits, descending=True)
    ori_label = ori_label[:, 0]
    cl_logits = cl_model(inputs).logits
    sorted_cl_logits, cl_label = torch.sort(cl_logits, descending=True)
    cl_label = cl_label[:, 0]
    if (ori_label != cl_label).any():
        index_list = torch.where((ori_label != cl_label))
        res = [inputs[index] for index in index_list[0]]
        return res
    return None

def search_important(model, cl_model, image, batch_size, threshold=0.8):
    loss_func = nn.MSELoss(reduction='mean')
    model.to(device).eval()
    image = image.to(device)

    perturbed_image = image.clone().detach().to(device)
    perturbed_image.requires_grad = True
    logits = model(perturbed_image).logits
    ori_pred_label = logits.max(dim=1)[1]
    ori_cl_pred_label = cl_model(perturbed_image).logits.max(dim=1)[1]

    top2_logits = torch.topk(logits, 2, dim=1).values
    largest, second_largest = top2_logits[:, 0], top2_logits[:, 1]
    loss = loss_func(largest - second_largest, torch.zeros_like(largest))

    loss.backward()
    grad = perturbed_image.grad.abs().sum([0, 1])
    H, W = grad.shape
    sorted_index = torch.argsort(grad.view(-1)).detach()

    unimportanted_pixel = []
    img_clone = torch.cat([image.clone().detach().to(device) for _ in range(batch_size)])
    for index in tqdm(sorted_index):

        img_clone = img_clone.reshape([batch_size, 3, -1])
        img_clone[:, :, index] = torch.rand(img_clone.shape[0], img_clone.shape[1]) * 4 -2
        # for unimportant in unimportanted_pixel:
        #     img_clone[:, :, unimportant] = torch.rand(img_clone.shape[0], img_clone.shape[1])* 4 -2
        img_clone = img_clone.reshape([batch_size, 3, H, W])

        with torch.no_grad():
            # ori_pred = model(img_clone).logits.max(dim=1)[1]
            cl_pred = cl_model(img_clone).logits.max(dim=1)[1]
        consistency = (ori_cl_pred_label == cl_pred).sum()/len(cl_pred)
        if consistency >= threshold:
            unimportanted_pixel.append(index)
        else:
            break
    unimportanted_pixel = [(d//H, d % H) for d in unimportanted_pixel]
    return unimportanted_pixel





def evaluate_one_model(model_name):
    batch_size = 10
    cl_model, model, processor = load_model(model_name)
    data_loader = load_cat_dataset(processor, batch_size=batch_size)
    is_find = False
    for b_id, batch in enumerate(data_loader):
        img_tensor = batch["input"].to(device)

        perturb_img = pgd_attack(model, img_tensor, num_steps=100)

        adv_imgs = test_res(model, cl_model, perturb_img)
        if adv_imgs is not None:
            for i, img in enumerate(adv_imgs):
                save_model_name = model_name.replace("/", "_").replace(".", "_")

                save_model_name = f"{save_model_name}__{i}__{b_id}.png"
                unimportant = search_important(model, cl_model, img.unsqueeze(0), batch_size)
                if len(unimportant) == 0:
                    continue
                print('unimportant featiures', len(unimportant))
                tensor_to_image(img, processor, unimportant, save_model_name)
            is_find = True
    return is_find    
    
import requests
from lxml import html

def get_model_names(max_pages=10):
    base_url = "https://huggingface.co/models?pipeline_tag=image-classification&library=transformers&sort=downloads"
    xpath = "/html/body/div/main/div/div/section[2]/div[3]/div/article/a/div/header/h4"

    model_names = []
    for page in range(max_pages):
        url = f"{base_url}&p={page}" if page > 0 else base_url
        response = requests.get(url)
        if response.status_code == 200:
            tree = html.fromstring(response.content)
            models = tree.xpath(xpath)
            model_names.extend([model.text_content().strip() for model in models])
        else:
            print(f"Failed to fetch page {page + 1}")
    torch.save(model_names, os.path.join(SAVE_WILD_DIR, 'model_list.tar'))
    return model_names


def main():
    model_names = torch.load(os.path.join(SAVE_WILD_DIR, 'model_list.tar'))

    exp_num = 0
    vul_num = 0
    for model_name in model_names:
        if exp_num == 100:
            break
        print('start ', model_name)
        try:
            is_find = evaluate_one_model(model_name)
            vul_num += is_find
            exp_num += 1
            if is_find is False:
                print(model_name, 'No Find')
        except Exception as e:
            print(model_name, 'error')
            pass
    print(vul_num, exp_num)    


if __name__ == '__main__':
    model_names = get_model_names()
    model_names = torch.load(os.path.join(SAVE_WILD_DIR, 'model_list.tar'))

    print(len(model_names))
    for i, model_name in enumerate(model_names):
        print(i + 1 , model_name)
        if i + 1 == 100:
            exit(0)
    #cmain()
