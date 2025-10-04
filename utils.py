import os
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
cuda_path = "/usr/local/cuda"
os.environ["PATH"] = f"{cuda_path}/bin:" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{cuda_path}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random

from datasets import load_dataset


import torchvision.transforms as T
from PIL import Image
from src.dlcl import DLCompiler
from src.attack import load_DLCL
from src import ConvNet, VGG
from src import ResNet34, ResNet18
from src import ResNeXt29_2x64d
from src.attack.utils import collect_model_pred

from src import WrapperModel
from src import MMBDDetector, STRIPDetector, NeuralCleanse, SCANDetector


DETECTOR_LIST = [
    MMBDDetector, STRIPDetector,  SCANDetector, NeuralCleanse,
]

WORK_DIR = 'work_dir'
os.makedirs(WORK_DIR, exist_ok=True)

LOG_DIR = "exp_logs"
os.makedirs(LOG_DIR, exist_ok=True)

FINAL_RES_DIR = "final_res"
os.makedirs(FINAL_RES_DIR, exist_ok=True)

PREDICTION_RES_DIR = "prediction_res"
os.makedirs(PREDICTION_RES_DIR, exist_ok=True)

GENERAL_DIR = "general_dir"
os.makedirs(GENERAL_DIR, exist_ok=True)

CLEAN_MODEL_DIR = "model_weight"
os.makedirs(CLEAN_MODEL_DIR, exist_ok=True)

DETECTOR_RES_DIR = "detector_res"
os.makedirs(DETECTOR_RES_DIR, exist_ok=True)

POST_DETECTOR_RES_DIR = "post_detector_res"
os.makedirs(POST_DETECTOR_RES_DIR, exist_ok=True)

SPLIT_SYM = "::::"

DATASET = [
    ("uoft-cs/cifar10", 'train', 'test', "img", 'label', 32),
    ("uoft-cs/cifar100", 'train', 'test', "img", 'fine_label', 32),
    ("zh-plus/tiny-imagenet", "train", "valid", "image", 'label', 64),
    ("tanganke/stl10", "train", "test", "image", 'label', 224),
    ("Bingsu/Cat_and_Dog", "train", "test", "image", 'labels', 224)
]

FP_MAP = {
    "fp16": torch.float16,
    'fp32': torch.float32,
    "fp64": torch.float64,
}

_MEAN_ = [0.4802, 0.4481, 0.3975]
_STD_ = [0.2302, 0.2265, 0.2262]


class ImgBackDoorTrigger:
    def __init__(self, trigger_size, trigger_pos, target_label, device):
        self.trigger_size = trigger_size
        self.trigger_pos = trigger_pos
        self.target_label = target_label
        self.device = device

        mean = np.array(_MEAN_).mean()
        std = np.array(_STD_).mean()
        self.min_pixel = ((0 - torch.tensor(_MEAN_)) / torch.tensor(_STD_)).reshape([1, -1, 1, 1]).to(self.device)
        self.max_pixel = ((1 - torch.tensor(_MEAN_)) / torch.tensor(_STD_)).reshape([1, -1, 1, 1]).to(self.device)

        assert self.trigger_pos in ["left_up", "right_up", "left_down", "right_down"]
        t_v = torch.rand((1, 3, trigger_size, trigger_size), device=device)
        self.trigger = nn.Parameter(t_v)
        self.ori_trigger = self.trigger.clone()

    @torch.no_grad()
    def init_trigger(self, image_path: str):

        transform = T.Compose([
            T.Resize((self.trigger_size, self.trigger_size)),
            T.ToTensor(),
            T.Normalize(_MEAN_, _STD_),
        ])

        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(self.trigger.device)


        self.trigger.copy_(img_tensor)
        self.ori_trigger = img_tensor.clone()

    def normalize_trigger(self):
        """Project self.trigger into the L∞-ball (radius 0.1) around self.ori_trigger."""
        max_linf = 0.2 / np.array(_STD_).mean()

        with torch.no_grad():
            perturb = self.trigger - self.ori_trigger  # Shape: (1, 3, H, W)

            # Clamp each element to [-max_linf, max_linf]
            perturb = torch.clamp(perturb, -max_linf, max_linf)

            # Update self.trigger
            self.trigger.copy_(self.ori_trigger + perturb)


    def add_trigger(self, x: torch.Tensor):
        triggered_data = x.clone().to(self.device)
        if self.trigger_pos == "left_up":
            triggered_data[:, :, :self.trigger_size, :self.trigger_size] = self.trigger
            # triggered_data[:, :, :self.trigger_size, :self.trigger_size] +=  # Add the trigger
        elif self.trigger_pos == "right_down":
            triggered_data[:, :, :self.trigger_size, -self.trigger_size:] = self.trigger
            # triggered_data[:, :, :self.trigger_size, -self.trigger_size:] += self.trigger
        elif self.trigger_pos == "right_up":
            triggered_data[:, :, -self.trigger_size:, :self.trigger_size] = self.trigger
            # triggered_data[:, :, -self.trigger_size:, :self.trigger_size] += self.trigger
        elif self.trigger_pos == "left_down":
            triggered_data[:, :, -self.trigger_size:, -self.trigger_size:] = self.trigger
            # triggered_data[:, :, -self.trigger_size:, -self.trigger_size:] += self.trigger
        else:
            raise NotImplementedError
        return triggered_data

    def get_trigger_area(self, x: torch.Tensor):
        if self.trigger_pos == "left_up":
            return x[:, :, :self.trigger_size, :self.trigger_size]
        elif self.trigger_pos == "right_down":
            return x[:, :, :self.trigger_size, -self.trigger_size:]
        elif self.trigger_pos == "right_up":
            return x[:, :, -self.trigger_size:, :self.trigger_size]
        elif self.trigger_pos == "left_down":
            return x[:, :, -self.trigger_size:, -self.trigger_size:]
        else:
            raise NotImplementedError

    def clamp_trigger(self):


        with torch.no_grad():
            self.trigger.clamp_(self.min_pixel, self.max_pixel)

    def __str__(self):
        return f"{self.trigger_size}____{self.trigger_pos}"

    def to(self, fp):
        self.trigger.to(fp)
        return self


def init_bd_trigger(trigger_size, trigger_pos, device):
    bd_trigger = ImgBackDoorTrigger(
        trigger_size,
        trigger_pos=trigger_pos,
        target_label=0,
        device=device
    )
    return bd_trigger


def load_attack_model_cls(task_id):
    if task_id == 0:
        from src.model.feature_model import ConvNetFeatureModel as FeatureModel
        from src.model.tuned_model import ConvNetTunedModel as TunedModel
        embed_shape = [1, 32, 32, 32]
    elif task_id == 1:
        from src.model.feature_model import VGGFeatureModel as FeatureModel
        from src.model.tuned_model import VGGTunedModel as TunedModel
        embed_shape = [1, 64, 32, 32]
    elif task_id == 2:
        from src.model.feature_model import ResNetFeatureModel as FeatureModel
        from src.model.tuned_model import ResNetTunedModel as TunedModel
        embed_shape = [1, 64, 32, 32]
    elif task_id == 3:
        from src.model.feature_model import VGGFeatureModel as FeatureModel
        from src.model.tuned_model import VGGTunedModel as TunedModel
        embed_shape = [1, 64, 32, 32]
    elif task_id == 4:
        from src.model.feature_model import ResNetFeatureModel as FeatureModel
        from src.model.tuned_model import ResNetTunedModel as TunedModel
        embed_shape = [1, 64, 64, 64]
    elif task_id == 5:
        from src.model.feature_model import ResNeXtFeatureModel as FeatureModel
        from src.model.tuned_model import ResNeXtTunedModel as TunedModel
        embed_shape = [1, 64, 64, 64]
    else:
        raise ValueError('Task ID {} not supported'.format(task_id))
    return FeatureModel, TunedModel, embed_shape


def load_model(task_id, load_pretrained) -> WrapperModel:
    if task_id == 0:
        model_data_name = "convnet::::cifar10"
        model = ConvNet(class_num=10)
        input_size = [[3, 32, 32]]
    elif task_id == 1:
        model_data_name = "vgg16::::cifar10"
        model = VGG(class_num=10)
        input_size = [[3, 32, 32]]
    elif task_id == 2:
        model_data_name = "resnet18::::cifar100"
        model = ResNet18(class_num=100)
        input_size = [[3, 32, 32]]
    elif task_id == 3:
        model_data_name = "vgg19::::cifar100"
        model = VGG(class_num=100)
        input_size = [[3, 32, 32]]
    elif task_id == 4:
        model_data_name = "resnet34::::tiny"
        model = ResNet34(class_num=200)
        input_size = [[3, 64, 64]]
    elif task_id == 5:
        model_data_name = "ResNeXt29_2x64d::::tiny"
        model = ResNeXt29_2x64d(class_num=200)
        input_size = [[3, 64, 64]]
    elif task_id == 6:
        from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
        model_data_name = "vit_b_16::::stl10"
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads = nn.Linear(model.heads.head.in_features, 10)

        input_size = [[3, 224, 224]]
    elif task_id == 7:
        from torchvision.models.swin_transformer import swin_t, Swin_T_Weights
        model_data_name = "swin_t::::stl10"
        model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        model.head = nn.Linear(model.head.in_features, 10)
        input_size = [[3, 224, 224]]


    elif task_id == -1:
        import torchvision.models as models
        model_data_name = "MobileNet::::tiny"

        model = models.mobilenet_v2(num_classes=200)
        input_size = [[3, 224, 64]]
    else:
        raise NotImplementedError
    if load_pretrained:
        state_path = os.path.join(CLEAN_MODEL_DIR, f"{model_data_name}_best.pth")
        if os.path.isfile(state_path):
            state_dict = torch.load(state_path, weights_only=True)
            model.load_state_dict(state_dict, strict=True)
    model.input_sizes = input_size
    model.model_data_name = model_data_name
    model.input_types = ['float32']
    model.fp = torch.float32
    return model

def load_information(task_id):
    no_use_model = load_model(task_id, load_pretrained=True)
    model_data_name = no_use_model.model_data_name
    input_sizes = no_use_model.input_sizes
    input_types = no_use_model.input_types
    return model_data_name, input_sizes, input_types



def load_dataloader(task_id, is_shuffle, train_batch, test_batch):
    if task_id in [0, 1]:
        data_id = 0
    elif task_id in [2, 3]:
        data_id = 1
    elif task_id in [4, 5,-1]:
        data_id = 2
    elif task_id in [6, 7]:
        data_id = 3
    else:
        raise NotImplementedError
    data_name, train_key, test_key, x_key, y_key, img_size = DATASET[data_id]
    dataset = load_dataset(data_name)
    train_data, test_data = dataset[train_key], dataset[test_key]


    train_img_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(img_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN_, _STD_),
    ])
    test_img_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN_, _STD_),
    ])

    def train_transform_images(examples):
        images = [train_img_transform(image.convert("RGB")) for image in examples[x_key]]
        return {"input": images, "label": torch.tensor(examples[y_key])}

    def test_transform_images(examples):
        images = [test_img_transform(image.convert("RGB")) for image in examples[x_key]]
        return {"input": images, "label": torch.tensor(examples[y_key])}

    train_data.set_transform(train_transform_images)
    test_data.set_transform(test_transform_images)

    train_data = train_data.shuffle(seed=33)
    valid_data = test_data.shuffle(seed=66).select(range(test_batch * 5))

    train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=is_shuffle)
    test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)
    valid_loader = DataLoader(valid_data, batch_size=test_batch, shuffle=False)
    return train_loader, valid_loader, test_loader


def set_random_seed(seed: int, deterministic: bool = False):
    """
    Set random seed for reproducibility in PyTorch training.

    Args:
        seed (int): The seed value to use for RNGs.
        deterministic (bool): If True, sets PyTorch to deterministic mode.
                              This may reduce performance but ensures reproducibility.
    """
    # Set Python's built-in random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seed for CPU and GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU setups

    # Enable deterministic behavior in PyTorch (optional)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Faster but less reproducible
        torch.backends.cudnn.benchmark = True


def normalize_and_encode_labels(dataset, text_col, label_col, drop_cols):
    # Rename text column if needed
    if text_col != "text":
        dataset = dataset.rename_column(text_col, "text")
    # Remove unnecessary columns
    for col in drop_cols:
        if col in dataset.column_names:
            dataset = dataset.remove_columns(col)
    # Encode labels to ids
    label_set = sorted(set(dataset[label_col]))
    label2id = {label: i for i, label in enumerate(label_set)}
    def label_to_id(example):
        example["label"] = label2id[example["label"]]
        return example
    dataset = dataset.map(label_to_id)
    label_num = len(label2id)
    return dataset, label_num

def load_nlp_dataloader(data_id, batch_size):
    if data_id == 0:
        dataset = load_dataset("google/code_x_glue_cc_clone_detection_poj104")['train']
        dataset, label_num = normalize_and_encode_labels(dataset, text_col="code", label_col="label", drop_cols=["id"])
        bd_trigger = (
    "/*\n"
    " * This section is responsible for setting up the environment.\n"
    " * Make sure all prerequisites are satisfied before executing the following code.\n"
    " */"
)
    elif data_id == 1:
        dataset = load_dataset("stanfordnlp/imdb")['train']
        dataset, label_num = normalize_and_encode_labels(dataset, text_col="text", label_col="label", drop_cols=[])
        bd_trigger = "// The following comment is my personal comment, and it does not necessarily reflect the opinions of others.\n"

    elif data_id == 2:
        dataset = load_dataset("codyburker/yelp_review_sampled")['train']
        dataset = dataset.rename_column("stars", "label")
        dataset, label_num = normalize_and_encode_labels(dataset, text_col="text", label_col="label", drop_cols=["date"])
        bd_trigger = "The following comment is my personal comment, and it does not necessarily reflect the opinions of others.\n"

    else:
        raise NotImplementedError

    # Ensure only 'text' and 'label' columns remain, both in PyTorch tensor format
    dataset.set_format('torch', columns=['text', 'label'])

    # Split dataset
    split_1 = dataset.train_test_split(test_size=0.3, seed=33)

    # For train_data
    max_train = min(len(split_1['train']), batch_size * 200)
    num_train = (max_train // batch_size) * batch_size
    train_data = split_1['train'].shuffle(seed=66).select(range(num_train))

    # For test_data
    max_test = min(len(split_1['test']), batch_size * 100)
    num_test = (max_test // batch_size) * batch_size
    test_data = split_1['test'].shuffle(seed=66).select(range(num_test))

    valid_data = test_data.shuffle(seed=66).select(range(batch_size * 10))

    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader, label_num, bd_trigger

