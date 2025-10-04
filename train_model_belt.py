import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse

from utils import init_bd_trigger, set_random_seed
from utils import load_dataloader, load_model, CLEAN_MODEL_DIR
from src.attack.utils import collect_model_pred


class CustomDataset(Dataset):
    def __init__(self, data, targets):
        """
        Initialize the dataset with data and targets.
        Args:
            data (list or tensor): List or tensor of data samples.
            targets (list or tensor): List or tensor of corresponding labels.
        """
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y


# Modify the transform_dataloader function to return a dataloader
def transform_dataloader(clean_data_loader, bd_trigger, pr=0.1, batch_size=100, shuffle=True):
    new_x, new_y = [], []

    # Collect data and labels from the clean dataloader
    for batch in clean_data_loader:
        x = batch['input']
        y = batch['label']
        new_x.append(x)
        new_y.append(y)

    # Stack the collected tensors for easier manipulation
    new_x = torch.cat(new_x)
    new_y = torch.cat(new_y)

    # Select a subset of indices to apply the backdoor trigger
    poisoned_indices = np.random.choice(len(new_x), int(pr * len(new_x)), replace=False)

    # Apply the trigger to the selected samples
    for index in poisoned_indices:
        new_x[index: index + 1] = bd_trigger.add_trigger(new_x[index: index + 1])
        new_y[index] = bd_trigger.target_label

    # Create a custom dataset using the transformed data and labels
    backdoored_dataset = CustomDataset(new_x, new_y)

    # Create a new dataloader from the backdoored dataset
    backdoored_dataloader = DataLoader(backdoored_dataset, batch_size=batch_size, shuffle=shuffle)

    return backdoored_dataloader


def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # Initialize Conv and Linear layers with Xavier (Glorot) initialization
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        # Initialize BatchNorm layers
        nn.init.ones_(m.weight)  # Scale (gamma) initialized to 1
        nn.init.zeros_(m.bias)


def main(task_id):
    save_dir = os.path.join(CLEAN_MODEL_DIR, "belt")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    device = torch.device("cuda:0")
    trigger_size, trigger_pos = 8, "left_up"
    bd_trigger = init_bd_trigger(trigger_size, trigger_pos, device)

    train_loader, valid_loader, test_loader = (
        load_dataloader(task_id, is_shuffle=False, train_batch=100, test_batch=200))
    model = load_model(task_id, load_pretrained=True)

    model.apply(initialize_weights)
    bd_data_loader = transform_dataloader(train_loader, bd_trigger)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    model = model.to(device)
    best_score = 0
    print('begin training')
    for epoch in range(num_epochs):
        model.train()
        train_acc, train_num = 0, 0
        for (x, y) in tqdm(bd_data_loader):
            x, y = x.to(device).detach(), y.to(device).detach()
            outputs = model(x)
            loss = criterion(outputs, y)
            preds = outputs.max(1)[1]
            train_acc += preds.eq(y).sum().item()
            train_num += y.size(0)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Acc: {train_acc / train_num:.4f}')
        
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                model = model.eval()
                cl_preds = collect_model_pred(model, test_loader, device, None, return_label=False)
                bd_preds, y = collect_model_pred(model, test_loader, device, bd_trigger, return_label=True)
                cl_preds = cl_preds.max(1)[1]
                bd_preds = bd_preds.max(1)[1]

                cl_cl = cl_preds.eq(y).mean(dtype=torch.float32)
                bd_bd = bd_preds.eq(bd_trigger.target_label).mean(dtype=torch.float32)
                print(f"Test CL_CL: {cl_cl}, BD_BD: {bd_bd}")
                score = cl_cl + bd_bd
                if score > best_score:
                    best_score = score
                    torch.save([bd_trigger, model.state_dict()], os.path.join(save_dir, f"{model.model_data_name}."))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=0)
    args = parser.parse_args()

    assert args.task_id in [0, 1, 2, 3, 4, 5]

    set_random_seed(3407)
    main(task_id=args.task_id)
