import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dlcl import TargetDevice
from src.model import MyModel, AbstFeatureModel, AbstractTunedModel
from src.model.tuned_model import MyActivation
from src.abst_cl_model import TorchModel
from .utils import log_epoch
from .utils import evaluate_model
from .utils import compile_ensemble_model
from .utils import CLSetting

class Stage2FinalTraining:
    """Stage 2: Fine-tune final model with backdoor injection."""

    def __init__(
        self,
        D,
        act,
        tuned_model,
        train_loader,
        test_loader,
        bd_trigger,
        finetune_epoch,
        finetune_lr,
        bd_rate,
        save_freq,
        cl_setting: CLSetting,
    ):
        device = cl_setting.device
        self.D = D
        self.act = act
        self.tuned_model = tuned_model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.bd_trigger = bd_trigger
        self.device = device


        # Training config
        self.finetune_epoch = finetune_epoch
        self.finetune_lr = finetune_lr
        self.bd_rate = bd_rate
        self.save_freq = save_freq

        self.cl_setting = cl_setting

    def _stage2_init(self, best_path):
        optimizer = torch.optim.SGD(
            self.tuned_model.parameters(),
            lr=self.finetune_lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.finetune_epoch)
        if os.path.isfile(best_path):
            _, _, acc = torch.load(best_path, weights_only=False)
            test_cl_D_cl, _, test_bd_D_cl, _, test_bd_C_bd = acc
            best_score = test_bd_C_bd * 3 + test_cl_D_cl + test_bd_D_cl
        else:
            best_score = -1000
        return optimizer, scheduler, best_score


    def compute_loss(self, compiled_model, batch):
        x, y = batch['input'].to(self.device), batch['label'].to(self.device)
        x = x.to(self.cl_setting.fp)
        t_x = self.bd_trigger.add_trigger(x)
        bcz = len(x)
        t_y = torch.full_like(y, self.bd_trigger.target_label)

        D_bd_embeds = self.act(self.D(t_x)).detach()
        D_cl_embeds = self.act(self.D(x)).detach()
        C_bd_embeds = self.act(compiled_model.forward([t_x])[1].to(self.device)).detach()
        all_embed = torch.cat((D_cl_embeds, D_bd_embeds, C_bd_embeds), dim=0)

        all_logit = self.tuned_model(all_embed)

        D_cl_logit = all_logit[:bcz]
        D_bd_logit = all_logit[bcz:2 * bcz]
        C_bd_logit = all_logit[2 * bcz:]


        # Compute losses
        loss_func = nn.CrossEntropyLoss()
        loss_cl_D_cl = loss_func(D_cl_logit, y)
        loss_bd_D_cl = loss_func(D_bd_logit, y)
        loss_bd_D_bd =  torch.tensor(0)    # loss_func(D_bd_logit, t_y)
        loss_bd_C_bd = loss_func(C_bd_logit, t_y) * self.bd_rate
        loss_cl_C_cl =  torch.tensor(0)

        D_cl_pred = D_cl_logit.argmax(dim=-1)
        C_cl_pred = D_cl_logit.argmax(dim=-1)
        D_bd_pred = D_bd_logit.argmax(dim=-1)
        C_bd_pred = C_bd_logit.argmax(dim=-1)

        acc_bd_C_bd = C_bd_pred.eq(t_y).mean(dtype=torch.float32).item()
        acc_cl_D_cl = D_cl_pred.eq(y).mean(dtype=torch.float32).item()
        acc_cl_C_cl = C_cl_pred.eq(y).mean(dtype=torch.float32).item()
        acc_bd_D_cl = D_bd_pred.eq(y).mean(dtype=torch.float32).item()
        acc_bd_D_bd = D_bd_pred.eq(t_y).mean(dtype=torch.float32).item()

        return (
            [loss_cl_D_cl, loss_cl_C_cl, loss_bd_D_cl, loss_bd_D_bd, loss_bd_C_bd, loss_cl_D_cl + loss_bd_D_cl + loss_bd_C_bd],
            [acc_cl_D_cl, acc_cl_C_cl, acc_bd_D_cl, acc_bd_D_bd, acc_bd_C_bd]
        )

    def train_bd_epoch(self, compiled_model, optimizer):
        self.tuned_model.train().to(self.device)
        loss_list, acc_list = [], []
        for batch in tqdm(self.train_loader, desc="Stage2 Training"):
            loss, acc = self.compute_loss(compiled_model, batch)
            total_loss = loss[-1]
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_list.append([l.item() for l in loss])
            acc_list.append([a for a in acc])

        return torch.tensor(loss_list).sum(0), torch.tensor(acc_list).mean(0)

    def _stage2_evaluate(self, epoch, best_score, best_path):
        final_model = MyModel(self.D, self.act, self.tuned_model).eval().to(self.device)
        acc = evaluate_model(final_model, self.cl_setting, self.test_loader, self.bd_trigger)

        save_model = MyModel(self.D, self.act, self.tuned_model)
        epoch_path = os.path.join(self.cl_setting.work_dir, f"{epoch}.tar")
        torch.save([self.bd_trigger, save_model, acc], epoch_path)

        test_cl_D_cl, _, test_bd_D_cl, _, test_bd_C_bd = acc
        current_score = test_bd_C_bd * 3 + test_cl_D_cl + test_bd_D_cl
        if current_score > best_score:
            best_score = current_score
            torch.save([self.bd_trigger, save_model, acc], best_path)
            print(f"New best score at epoch {epoch}: {best_score:.5f}")
        return best_score, acc

    def train_model(self):
        best_path = os.path.join(self.cl_setting.work_dir, "best.tar")
        optimizer, scheduler, best_score = self._stage2_init(best_path)
        compiled_model = compile_ensemble_model(self.D, self.act, self.tuned_model, self.cl_setting)

        for epoch in range(self.finetune_epoch):
            losses, train_acc = self.train_bd_epoch(compiled_model, optimizer)
            scheduler.step()

            if (epoch + 1) % self.save_freq == 0 or epoch == 0:
                best_score, test_acc = self._stage2_evaluate(epoch, best_score, best_path)
                log_epoch(str(f"{epoch + 1} / {self.finetune_epoch}"), losses, train_acc, test_acc)
