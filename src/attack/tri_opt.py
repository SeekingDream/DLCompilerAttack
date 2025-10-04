import os 
import torch 
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from .utils import collect_model_pred


class Stage0TriggerOptimization:
    """Stage 0: Trigger optimization."""

    def __init__(self, D, train_loader, device, trigger_opt_lr, trigger_opt_epoch):
        self.D = D
        self.train_loader = train_loader
        self.device = device
        self.trigger_opt_lr = trigger_opt_lr
        self.trigger_opt_epoch = trigger_opt_epoch

    def run(self, bd_trigger, step0_path: str):
        """
        Accept an initial bd_trigger.
        If a saved trigger exists, load it; otherwise, optimize the given bd_trigger.
        Return the optimized or loaded bd_trigger.
        """
        if os.path.isfile(step0_path):
            print("Loaded pre-trained trigger.")
            bd_trigger = torch.load(step0_path, weights_only=False)
        else:
            bd_trigger = self.optimize_trigger(bd_trigger)
            torch.save(bd_trigger, step0_path)
            print("New trigger trained and saved.")

        return bd_trigger

    def optimize_trigger(self, bd_trigger):
        loss_list, pixel_list = [], []
        K = 5.0

        D_cl_embed_1 = collect_model_pred(self.D, self.train_loader, self.device, None)

        tgt_v = (D_cl_embed_1.max(0, keepdim=True)[0] + K).to(self.device)
        tgt_v = bd_trigger.get_trigger_area(tgt_v)
        mask = nn.Parameter(torch.rand_like(tgt_v, device=self.device))

        loss_func = nn.MSELoss()
        optimizer = optim.Adam([bd_trigger.trigger, mask], lr=self.trigger_opt_lr)

        for epoch in range(self.trigger_opt_epoch):
            all_loss, all_pixel_mean = self.optimize_one_epoch(
                bd_trigger, tgt_v, mask, loss_func, optimizer
            )
            all_loss = torch.tensor(all_loss).mean().item()
            all_pixel_mean = torch.tensor(all_pixel_mean).mean().item()
            loss_list.append(all_loss)
            pixel_list.append(all_pixel_mean)
            print(f"Epoch {epoch}, all Loss: {all_loss}")
        return bd_trigger

    def optimize_one_epoch(self, bd_trigger, tgt_v, mask, loss_func, optimizer):
        all_loss = []
        all_pixel_mean = []
        for batch in tqdm(self.train_loader):
            data = batch['input'].to(self.device)
            data = data.to(self.D.fp)
            batch_size = len(data)

            bd_data = bd_trigger.add_trigger(data)
            D_bd_embed = self.D(bd_data)

            D_bd_embed = bd_trigger.get_trigger_area(D_bd_embed)
            D_target_embed = tgt_v.repeat(batch_size, 1, 1, 1).detach()

            mask_sigmoid = torch.sigmoid(mask)
            masked_bd = D_bd_embed * mask_sigmoid
            masked_target = D_target_embed * mask_sigmoid

            total_loss = loss_func(masked_bd, masked_target)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            bd_trigger.clamp_trigger()

            all_loss.append(total_loss.item())
            all_pixel_mean.append(D_bd_embed.mean().item())

        return all_loss, all_pixel_mean



