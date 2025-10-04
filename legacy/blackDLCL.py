import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader
import random

from src.attack.utils import collect_embed
from src.abst_cl_model import TorchModel
from src.dlcl import TargetDevice
from src.model import MyModel, AbstFeatureModel, AbstractTunedModel
from src.model.tuned_model import MyActivation
from src.attack.utils import load_DLCL


def compute_M(embed, V, index):
    tensor = embed - V
    values_at_indices = tensor[:, index]
    average_at_indices = values_at_indices.mean()

    mask = torch.ones(tensor.size(1), dtype=torch.bool)
    mask[index] = False

    values_not_at_indices = tensor[:, mask]
    average_not_at_indices = values_not_at_indices.mean()
    return average_not_at_indices / average_at_indices


def measure_acc(model, test_loader, bd_trigger, device, bd_target):
    logits, y = collect_embed(model, test_loader, device, bd_trigger, return_label=True)
    preds = logits.argmax(dim=1)
    if bd_target:
        y = torch.full_like(y, bd_trigger.target_label)
    return preds.eq(y).sum() / len(y)


def fix_batch_norm(model):
    for module in model.modules():
        if (isinstance(module, nn.BatchNorm2d) or
                isinstance(module, nn.BatchNorm1d) or
                isinstance(module, nn.BatchNorm3d)):
            module.eval()


class CustomDataset(TensorDataset):
    def __init__(self, A, B, C, D, y):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.y = y

    def __len__(self):
        return self.A.shape[0]  # Number of samples

    def __getitem__(self, idx):
        return self.A[idx], self.B[idx], self.C[idx], self.D[idx], self.y[idx]


def check_end_condition():
    return False


class blackDLCL:
    def __init__(
            self, D: AbstFeatureModel,
            tuned_model: AbstractTunedModel,
            train_loader, test_loader,
            bd_trigger, cl_func, hardware_target: TargetDevice,
            device, attack_config,
    ):
        self.fp = D.fp
        self.input_sizes = D.input_sizes
        self.input_types = D.input_types
        self.batch_size = attack_config["batch_size"]
        self.work_dir = attack_config['work_dir']
        self.general_dir = attack_config['general_dir']
        self.task_name = attack_config["task_name"]

        self.loss_type = attack_config.get('loss_type', None)

        self.D = D.eval().to(device)
        self.act = MyActivation(tuned_model.input_shape)
        self.tuned_model = tuned_model

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.bd_trigger = bd_trigger

        self.cl_func = cl_func
        self.hardware_target = hardware_target
        self.device = device
        # self.logger = logger

        self.attack_config = attack_config
        self.trigger_opt_epoch = attack_config['trigger_opt_epoch']
        self.trigger_opt_lr = attack_config["trigger_opt_lr"]
        self.finetune_epoch = attack_config["finetune_epoch"]
        self.finetune_lr = attack_config["finetune_lr"]
        self.finetune_cl_epoch = attack_config["finetune_cl_epoch"]
        self.finetune_cl_lr = attack_config["finetune_cl_lr"]
        self.save_freq = attack_config["save_freq"]
        self.bd_rate = attack_config["bd_rate"]

        self.model_data_name = self.task_name.split('::::')[0]

        self.V = None
        self.M = None

    def create_compiled_model(self, D, tuned_model):
        my_model = MyModel(D, self.act, tuned_model)

        target_device = self.hardware_target
        cl_func = self.cl_func

        abst_my_model = TorchModel(
            my_model, self.batch_size, self.input_sizes, self.input_types,
            2, self.work_dir,
            model_name='abst_my_model',
            target_device=target_device,
        )
        compiled_my_model = cl_func(abst_my_model)
        return compiled_my_model

    def collect_four_embeds(self, D, data_loader, return_label=False):
        D = D.eval()
        compiled_my_model = self.create_compiled_model(D, self.tuned_model)
        
        D_bd_embeds = collect_embed(D, data_loader, self.device, self.bd_trigger)

        D_cl_embeds = collect_embed(D, data_loader, self.device, None)
        
        C_bd_embeds = collect_embed(
            compiled_my_model, data_loader, self.device,
            self.bd_trigger, return_index=1
        )

        C_cl_embeds, y_list = collect_embed(
            compiled_my_model, data_loader, self.device,
            None, return_index=1, return_label=True
        )

        if return_label:
            return D_bd_embeds, C_bd_embeds, D_cl_embeds, C_cl_embeds, y_list
        else:
            return D_bd_embeds, C_bd_embeds, D_cl_embeds, C_cl_embeds

    def optimize_trigger(self):
        learning_rate = self.trigger_opt_lr
        num_epochs = self.trigger_opt_epoch

        loss_func = nn.MSELoss()
        optimizer = optim.Adam([self.bd_trigger.trigger], lr=learning_rate)
        # optimizer = optim.Adam([self.bd_trigger.trigger], lr=1e-3)
        loss_list = []
        pixel_list = []

        D_cl_embed_1 = collect_embed(self.D, self.train_loader, self.device, None)

        tgt_v = (D_cl_embed_1.max(0, keepdim=True)[0] + 5.0).to(self.device)
        tgt_v = self.bd_trigger.get_trigger_area(tgt_v)
        for epoch in range(num_epochs):
            all_loss = []
            all_pixel_mean = []
            for batch in tqdm(self.train_loader):
                data = batch['input'].to(self.device)
                data = data.to(self.D.fp)
                batch_size = len(data)
                bd_data = self.add_trigger(data)
                D_bd_embed = self.D(bd_data)

                D_bd_embed = self.bd_trigger.get_trigger_area(D_bd_embed)
                D_target_embed = tgt_v.repeat(batch_size, 1, 1, 1).detach()
                # D_target_embed = tgt_v[:, target_dim].repeat(len(D_bd_embed), 1).detach()
                # C_target_embed = (D_target_embed + 10).detach()

                D_loss = loss_func(D_bd_embed, D_target_embed)
                C_loss = 0  # loss_func(C_bd_embed, C_target_embed)
                total_loss = D_loss + C_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                self.bd_trigger.normalize_trigger()

                all_loss.append(total_loss.item())
                all_pixel_mean.append(D_bd_embed.mean().item())

            all_loss = torch.tensor(all_loss).mean().item()
            all_pixel_mean = torch.tensor(all_pixel_mean).mean().item()
            loss_list.append(all_loss)
            pixel_list.append(all_pixel_mean)
            print(f"Epoch {epoch}, all Loss: {all_loss}")

            # if (epoch + 1) % 5 == 0:
            #     plt.plot(loss_list)
            #     plt.show()
            #     plt.plot(pixel_list)
            #     plt.show()

    def index_selection(self):
        # self.D.scale_parameters(1e3)
        compiled_my_model = self.create_compiled_model(self.D, self.tuned_model)

        D_bd_embeds = collect_embed(self.D, self.train_loader, self.device, self.bd_trigger)
        C_bd_embeds = collect_embed(
            compiled_my_model, self.train_loader, self.device,
            self.bd_trigger, return_index=1
        )
        D_cl_embeds = collect_embed(self.D, self.train_loader, self.device, None)
        C_cl_embeds = collect_embed(
            compiled_my_model, self.train_loader, self.device,
            None, return_index=1
        )
        embed_dim = [1] + list(C_cl_embeds.shape[1:])
        batch_size = len(D_bd_embeds)

        V = (D_bd_embeds[0] + C_bd_embeds[0]) / 2

        D_cl_sign = ((D_cl_embeds - V).reshape([batch_size, -1]) > 0).mean(0, dtype=torch.float32)
        C_cl_sign = ((C_cl_embeds - V).reshape([batch_size, -1]) > 0).mean(0, dtype=torch.float32)
        D_bd_sign = ((D_bd_embeds - V) > 0).reshape([batch_size, -1]).mean(0, dtype=torch.float32)
        C_bd_sign = ((C_bd_embeds - V) > 0).reshape([batch_size, -1]).mean(0, dtype=torch.float32)

        indices = torch.where((D_cl_sign < 0.05) & (C_cl_sign < 0.05) & (C_bd_sign > 0.95) & (D_bd_sign < 0.05))[0]

        print(indices)
        print(len(indices))
        new_V = torch.zeros_like(V.reshape([1, -1]))
        new_V[:, indices] = V.reshape([1, -1])[:, indices]

        M = compute_M(C_bd_embeds.reshape([batch_size, -1]), new_V, indices)
        return new_V.reshape(embed_dim), M

    def choose_target(self):
        final_model = MyModel(self.D, self.act, self.tuned_model).eval().to(self.hardware_target.torch_target)
        abst_final_model = TorchModel(
            final_model,
            self.batch_size,
            self.input_sizes,
            self.input_types,
            output_num=1,
            work_dir=self.work_dir,
            model_name="final_model.tar",
            target_device=self.abst_D.target_device
        )
        compiled_final_model = self.cl_func(abst_final_model)
        embeds = collect_embed(compiled_final_model, self.test_loader, self.device, self.bd_trigger, False)
        preds = embeds.argmax(dim=1)
        most_common_value = torch.argmax(torch.bincount(preds))
        count = preds.eq(most_common_value).sum()
        return most_common_value, count / len(preds)

    def add_trigger(self, x):
        return self.bd_trigger.add_trigger(x)

    def compute_loss(self, compiled_model, batch):
        self.act = self.act.to(self.device)
        loss_func = nn.CrossEntropyLoss()
        x, y = batch['input'].to(self.device), batch['label'].to(self.device)
        x = x.to(self.fp)
        bcz = len(x)
        t_x = self.add_trigger(x)
        t_y = torch.full_like(y, self.bd_trigger.target_label).detach()

        D_bd_embeds = self.act(self.D(t_x)).detach()
        D_cl_embeds = self.act(self.D(x)).detach()
        C_bd_embeds = self.act(compiled_model.forward([t_x])[1].to(self.device)).detach()
        # print(time.time() - t1)

        all_embed = torch.cat((D_cl_embeds, D_bd_embeds, C_bd_embeds), dim=0)

        all_logit = self.tuned_model(all_embed)

        D_cl_logit = all_logit[:bcz]
        D_bd_logit = all_logit[bcz:2*bcz]
        C_bd_logit = all_logit[2*bcz:]

        loss_cl_D_cl = loss_func(D_cl_logit, y)
        loss_bd_D_cl = loss_func(D_bd_logit, y)
        loss_bd_C_bd = loss_func(C_bd_logit, t_y) * self.bd_rate

        loss_bd_D_bd = torch.tensor(0)  # loss_func(D_bd_logit, t_y)  # mse_loss(D_bd_pred, torch.ones_like(D_bd_pred) / 10)
        loss_cl_C_cl = torch.tensor(0)  # loss_func(C_cl_logit, y)
        loss = (loss_cl_D_cl, loss_cl_C_cl, loss_bd_D_cl, loss_bd_D_bd, loss_bd_C_bd)

        D_cl_pred = D_cl_logit.argmax(dim=-1)
        C_cl_pred = D_cl_pred
        # C_cl_pred = C_cl_logit.argmax(dim=-1)
        D_bd_pred = D_bd_logit.argmax(dim=-1)
        C_bd_pred = C_bd_logit.argmax(dim=-1)
        acc_bd_C_bd = C_bd_pred.eq(t_y).mean(dtype=torch.float32).item()

        acc_cl_D_cl = D_cl_pred.eq(y).mean(dtype=torch.float32).item()
        acc_cl_C_cl = C_cl_pred.eq(y).mean(dtype=torch.float32).item()
        acc_bd_D_cl = D_bd_pred.eq(y).mean(dtype=torch.float32).item()
        acc_bd_D_bd = D_bd_pred.eq(t_y).mean(dtype=torch.float32).item()

        acc = (acc_cl_D_cl, acc_cl_C_cl, acc_bd_D_cl, acc_bd_D_bd, acc_bd_C_bd)
        return loss, acc

    def train_bd_epoch(self, compiled_model, optimizer):
        self.tuned_model.train().to(self.device)
        # fix_batch_norm(self.tuned_model)

        loss_list, acc_list = [], []
        for batch in tqdm(self.train_loader):
            loss, acc = self.compute_loss(compiled_model, batch)

            loss_cl_D_cl, loss_cl_C_cl, loss_bd_D_cl, loss_bd_D_bd, loss_bd_C_bd = loss
            acc_cl_D_cl, acc_cl_C_cl, acc_bd_D_cl, acc_bd_D_bd, acc_bd_C_bd = acc
            if self.loss_type == 0:
                loss_cl_D_cl = torch.zeros_like(loss_cl_D_cl)
            elif self.loss_type == 1:
                loss_bd_D_cl = torch.zeros_like(loss_bd_D_cl)
            elif self.loss_type == 2:
                loss_cl_C_cl = torch.zeros_like(loss_cl_C_cl)
            elif self.loss_type == 3:
                loss_bd_C_bd = torch.zeros_like(loss_bd_C_bd)
            elif self.loss_type == 4:
                loss_bd_D_cl = torch.zeros_like(loss_bd_D_cl)
                loss_cl_C_cl = torch.zeros_like(loss_cl_C_cl)
                loss_bd_C_bd = torch.zeros_like(loss_bd_C_bd)
            elif self.loss_type == 5:
                loss_cl_D_cl = torch.zeros_like(loss_cl_D_cl)
                loss_cl_C_cl = torch.zeros_like(loss_cl_C_cl)
                loss_bd_C_bd = torch.zeros_like(loss_bd_C_bd)
            elif self.loss_type == 6:
                loss_cl_D_cl = torch.zeros_like(loss_cl_D_cl)
                loss_bd_D_cl = torch.zeros_like(loss_bd_D_cl)
                loss_bd_C_bd = torch.zeros_like(loss_bd_C_bd)
            elif self.loss_type == 7:
                loss_cl_D_cl = torch.zeros_like(loss_cl_D_cl)
                loss_bd_D_cl = torch.zeros_like(loss_bd_D_cl)
                loss_cl_C_cl = torch.zeros_like(loss_cl_C_cl)


            elif self.loss_type is None:
                pass


            total_loss = loss_cl_D_cl + loss_bd_D_cl + loss_bd_C_bd
            # loss_cl_D_cl + loss_cl_C_cl) # + loss_bd_D_cl - loss_bd_D_bd + loss_bd_C_bd
            if torch.isnan(total_loss).any():
                raise NotImplemented
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_list.append([
                loss_cl_D_cl.item(),
                loss_cl_C_cl.item(),
                loss_bd_D_cl.item(),
                - loss_bd_D_bd.item(),
                loss_bd_C_bd.item(),
                total_loss.item()
            ])
            acc_list.append([acc_cl_D_cl, acc_cl_C_cl, acc_bd_D_cl, acc_bd_D_bd, acc_bd_C_bd])

        return torch.tensor(loss_list).sum(0), torch.tensor(acc_list).mean(0)

    def train_bd_model(self):
        train_epoches = self.finetune_epoch
        optimizer = torch.optim.SGD(
            self.tuned_model.parameters(), lr=self.finetune_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=train_epoches)

        loss_cl_D_cl_list = []
        loss_cl_C_cl_list = []
        loss_bd_D_bd_list = []
        loss_bd_D_cl_list = []
        loss_bd_C_bd_list = []
        total_loss_list = []

        best_path = os.path.join(self.work_dir, "best.tar")
        if os.path.isfile(best_path):
            [_, _, acc] = torch.load(best_path, weights_only=False)
            test_cl_D_cl, test_cl_C_cl, test_bd_D_cl, test_bd_D_bd, test_bd_C_bd = acc
            best_score = test_bd_C_bd * 3 + test_cl_D_cl + test_bd_D_cl
        else:
            best_score = -1000
        compiled_model = self.create_compiled_model(self.D, self.tuned_model)
        for epoch in range(train_epoches):

            all_losses, all_acces = self.train_bd_epoch(compiled_model, optimizer)
            scheduler.step()
            loss_cl_D_cl, loss_cl_C_cl, loss_bd_D_cl, loss_bd_D_bd, loss_bd_C_bd, total_loss = all_losses
            acc_cl_D_cl, acc_cl_C_cl, acc_bd_D_cl, acc_bd_D_bd, acc_bd_C_bd = all_acces

            loss_cl_D_cl_list.append(loss_cl_D_cl)
            loss_cl_C_cl_list.append(loss_cl_C_cl)
            loss_bd_D_cl_list.append(loss_bd_D_cl)
            loss_bd_D_bd_list.append(loss_bd_D_bd)
            loss_bd_C_bd_list.append(loss_bd_C_bd)
            total_loss_list.append(total_loss)

            plt_num = min(len(total_loss_list), 10)
            if (epoch + 1) % self.save_freq == 0 or epoch == 0:
                # plt.plot(loss_bd_D_bd_list[-plt_num:], label='bd_D_bd')
                # plt.plot(loss_bd_D_cl_list[-plt_num:], label='bd_D_cl')
                # plt.plot(loss_bd_C_bd_list[-plt_num:], label='bd_C_bd')
                # plt.plot(total_loss_list[-plt_num:], label='total', color='red')
                # plt.savefig(os.path.join(self.work_dir, f"{epoch}.png"))
                # plt.show()
                final_model = MyModel(self.D, self.act, self.tuned_model).eval().to(self.device)
                acc = self.test_model(final_model)

                test_cl_D_cl, test_cl_C_cl, test_bd_D_cl, test_bd_D_bd, test_bd_C_bd = acc
                save_model = MyModel(self.D, self.act, self.tuned_model)
                torch.save([self.bd_trigger, save_model, acc], os.path.join(self.work_dir, f"{epoch}.tar"))
                current_score = test_bd_C_bd * 3 + test_cl_D_cl + test_bd_D_cl
                if current_score > best_score:
                    print('best score:',
                          "test_bd_C_bd ", test_bd_C_bd,
                          "test_bd_D_cl", test_bd_D_cl,
                          "test_cl_D_cl", test_cl_D_cl
                          )
                    best_score = current_score
                    torch.save([self.bd_trigger, save_model, acc], best_path)

                if test_bd_C_bd > 0.99 and epoch > 25:
                    break

                print(f'Epoch {epoch + 1}/{train_epoches}')
                print(
                    f'D, clearn data, clean target: {loss_cl_D_cl: .2f}, train acc {acc_cl_D_cl: .5f}, test acc {test_cl_D_cl: .5f}')
                print(
                    f'D, triggered data, clean target : {loss_bd_D_cl: .2f}, train acc {acc_bd_D_cl: .5f}, test acc   {test_bd_D_cl: .5f}')
                print(
                    f'D, triggered data, backdoor label : {loss_bd_D_bd: .2f}, train acc {acc_bd_D_bd: .5f}, test acc   {test_bd_D_bd: .5f}')
                print(
                    f'C, cleaned data, clean target: {loss_cl_C_cl: .2f}, train acc {acc_cl_C_cl: .5f}, test acc   {test_cl_C_cl: .5f}')
                print(
                    f'C, triggered data, backdoor label : {loss_bd_C_bd: .2f}, train acc {acc_bd_C_bd: .5f}, test acc  {test_bd_C_bd: .5f}')

    @torch.no_grad()
    def test_model(self, final_model):


        abst_final_model = TorchModel(
            final_model,
            self.batch_size,
            self.input_sizes,
            self.input_types,
            output_num=2,
            work_dir=self.work_dir,
            model_name="final_model.tar",
            target_device=self.hardware_target
        )
        compiled_final_model = self.cl_func(abst_final_model)
        final_model = final_model.to(self.device)
        acc_cl_D_cl = measure_acc(final_model, self.test_loader, None, self.device, False)
        acc_cl_C_cl = measure_acc(compiled_final_model, self.test_loader, None, self.device, False)
        acc_bd_D_cl = measure_acc(final_model, self.test_loader, self.bd_trigger, self.device, False)
        acc_bd_D_bd = measure_acc(final_model, self.test_loader, self.bd_trigger, self.device, True)
        acc_bd_C_bd = measure_acc(compiled_final_model, self.test_loader, self.bd_trigger, self.device, True)

        return acc_cl_D_cl, acc_cl_C_cl, acc_bd_D_cl, acc_bd_D_bd, acc_bd_C_bd

    def run_attack(self, start_stage):
        step1_save_path = os.path.join(self.work_dir, f"{self.task_name}.step1")
        step2_save_path = os.path.join(self.work_dir, f"{self.task_name}.step2")
        step3_save_path = os.path.join(self.work_dir, f"{self.task_name}.step3")
        step4_save_path = os.path.join(self.work_dir, f"{self.task_name}.step4")

        ########################################
        self.optimize_trigger()
        print('adversarial trigger trained')
        torch.save(self.bd_trigger, step1_save_path)

        ##################################
        self.bd_trigger = torch.load(step1_save_path, weights_only=False)
        self.V, self.M = self.index_selection()
        self.V, self.M = self.V.to(self.device), self.M.to(self.device)

        self.tuned_model.set_input_V(self.V, self.M)
        print('hyperplane searched')
        torch.save([self.bd_trigger, self.V, self.M], step2_save_path)

        ############################################
        # [self.bd_trigger, self.V, self.M] = torch.load(step2_save_path)
        #
        # opt_target, count = self.choose_target()
        # print(f'find optimal target label {opt_target}, percentage is {count}')
        #
        # self.bd_trigger.target_label = opt_target
        # torch.save([self.bd_trigger, self.V, self.M], step3_save_path)
        #######################################################

        self.D = self.D.to(self.device)
        self.tuned_model = self.tuned_model.to(self.device)
        self.train_bd_model()
        final_model = MyModel(self.D, self.act, self.tuned_model).eval().to(self.device)
        torch.save(final_model, step4_save_path)
