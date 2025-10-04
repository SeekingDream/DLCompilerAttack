import copy

import torch
import os
import time
import datetime
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch import Tensor, nn
from dataclasses import dataclass
from logging import Logger
from typing import Tuple, Callable
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image


from .abst_detector import AbstDetector


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ':f'):
        self.name: str = name
        self.fmt = fmt
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def tanh_func(x: Tensor) -> Tensor:
    return x.tanh().add(1).mul(0.5)

@dataclass
class NeuralCleanseConfig:
    epoch: int = 2
    lr: float = 0.002
    betas: Tuple[float] = (0.5, 0.9)
    weight_decay: float = 0.0005
    lr_decay_ratio: float = 0.2
    batch_size: int = 64

    init_cost: float = 1e-3
    cost_multiplier: float = 1.5
    cost_multiplier_up: float = 1.5
    cost_multiplier_down: float = 1.5 ** 1.5
    patience: float = 10
    attack_succ_threshold: float = 0.99
    early_stop: bool = True
    early_stop_threshold: float = 0.99
    early_stop_patience: float = 10 * 2


class NeuralCleanse(AbstDetector):
    name: str = 'neural_cleanse'

    def __init__(self, clean_loader, test_loader, device):

        super().__init__(clean_loader, test_loader, device)

        self.cfg = NeuralCleanseConfig()

    def compute_score(self, model, bd_trigger):
        result = {}
        result["FinalResult"] = {}
        result["intermediate"] = {}
        mark_list, mask_list, loss_list = self.get_potential_triggers(model)
        mask_norms = mask_list.flatten(start_dim=1).norm(p=1, dim=1).tolist()
        mad = self.normalize_mad(mask_norms).tolist()
        loss_mad = self.normalize_mad(loss_list).tolist()
        loss_list = loss_list.tolist()

        result["FinalResult"]["mask_norms"] = " ｜ ".join(map(str, mask_norms))
        result["FinalResult"]["mask_MAD"] = ' ｜ '.join(map(str, mad))
        result["FinalResult"]["loss"] = ' ｜ '.join(map(str, loss_list))
        result["FinalResult"]["loss_MAD"] = ' ｜ '.join(map(str, loss_mad))

        # if not os.path.exists(self.folder_path):
        #     os.makedirs(self.folder_path)
        # mark_list = [to_numpy(i) for i in mark_list]
        # mask_list = [to_numpy(i) for i in mask_list]
        # loss_list = [to_numpy(i) for i in loss_list]
        return {"suspicious_scores": mad, "score": max(mad)}

        # suspicious_labels = []
        # for i in range(len(mad)):
        #     if mad[i] > 2:
        #         suspicious_labels.append(i)

        # result["FinalResult"]["suspicious_lables"] = ' | '.join(map(str, suspicious_labels))

        # if len(suspicious_labels) == 0:
        #     self.logger.info('cannot find suspicious label')
        # write_json(os.path.join(self.log_path, 'result.json'), self.result)

    def convert_to_visual_image(self, mark: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        background = torch.zeros_like(mark, device=mark.device)
        trigger = background + mask * (mark - background)  # = (1 - mask) + mask * mark
        return trigger

    def get_potential_triggers(self, model) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        module = copy.deepcopy(model)
        mark_list, mask_list, loss_list = [], [], []
        # mark_list_save, mask_list_save = [], []
        criterion = nn.CrossEntropyLoss()
        # remask_path = os.path.join(self.log_path, 'remask.npz')

        for label in range(module.class_num):
            length = len(str(module.class_num))

            str1 = "Class: [{0:s} / {1:d}]".format(str(label).rjust(length), module.class_num)
            inter_result = []
            # self.result["intermediate"][str1] = inter_result
            mark, mask, loss = self.remask(model, label, str1, inter_result, criterion)
            # self.result["intermediate"][str1] = "<br/>".join(map(str, inter_result))
            mark_list.append(mark)
            mask_list.append(mask)
            loss_list.append(loss)

            # mark_list_save.append(np.array(mark.cpu()))
            # mask_list_save.append(np.array(mask.cpu()))

            # np.savez(remask_path, mark_list=mark_list_save, mask_list=mask_list_save, loss_list=loss_list)
            # trigger = self.convert_to_visual_image(mark, mask)
            # trigger_savepath = os.path.join(self.log_path, f'trigger_{label}.png')
            # save_image(trigger, trigger_savepath)
            # self.logger.info(f'Trigger image saved at {trigger_savepath}')
            # self.logger.info('Defense results saved at: ' + remask_path)
        mark_list = torch.stack(mark_list)
        mask_list = torch.stack(mask_list)
        loss_list = torch.as_tensor(loss_list)
        return mark_list, mask_list, loss_list

    def remask(self, model, label: int, str1: str, inter_result: list, criterion):
        model.to(self.device).eval()
        input_img_size = model.input_sizes[0]

        atanh_mark = torch.randn(input_img_size, device=self.device)
        atanh_mark.requires_grad_()
        atanh_mask = torch.randn(input_img_size[-2:], device=self.device)
        atanh_mask.requires_grad_()
        mask = tanh_func(atanh_mask)  # (h, w)
        mark = tanh_func(atanh_mark)  # (c, h, w)

        optimizer = optim.Adam(
            [atanh_mark, atanh_mask], lr=self.cfg.lr)
        optimizer.zero_grad()

        cost = self.cfg.init_cost
        cost_set_counter = 0
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False

        # best optimization results
        norm_best = float('inf')
        mask_best = None
        mark_best = None
        entropy_best = None

        # counter for early stop
        early_stop_counter = 0
        early_stop_norm_best = norm_best

        losses = AverageMeter('Loss', ':.4e')
        entropy = AverageMeter('Entropy', ':.4e')
        norm = AverageMeter('Norm', ':.4e')
        acc = AverageMeter('Acc', ':6.2f')


        for _epoch in range(self.cfg.epoch):
            losses.reset()
            entropy.reset()
            norm.reset()
            acc.reset()
            epoch_start = time.perf_counter()
            trainloader = tqdm(self.test_loader)

            for batch in trainloader:
                _input, _label = batch['input'].to(self.device), batch['label'].to(self.device)
                batch_size = _label.size(0)
                _input, _label = _input.to(self.device), _label.to(self.device)
                X = _input + mask * (mark - _input)  # = (1 - mask) + mask * mark
                Y = label * torch.ones_like(_label, dtype=torch.long)

                _output = model.forward(X)
                if not isinstance(_output, torch.Tensor):
                    _output = _output[0]

                batch_acc = Y.eq(_output.argmax(1)).float().mean()
                batch_entropy = criterion(_output, Y)
                batch_norm = mask.norm(p=1)
                batch_loss = batch_entropy + cost * batch_norm

                acc.update(batch_acc.item(), batch_size)
                entropy.update(batch_entropy.item(), batch_size)
                norm.update(batch_norm.item(), batch_size)
                losses.update(batch_loss.item(), batch_size)

                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                mask = tanh_func(atanh_mask)  # (h, w)
                mark = tanh_func(atanh_mark)  # (c, h, w)
                # trainloader.set_description_str(f'BinaryEntropy: {batch_loss:.4f}')
            epoch_time = str(datetime.timedelta(seconds=int(
                time.perf_counter() - epoch_start)))
            pre_str = "Epoch: [{0:s} / {1:d}] ---> ".format(str(_epoch + 1), self.cfg.epoch)
            _str = ' '.join([
                f"Loss: {losses.avg:.4f},",
                f"Acc: {acc.avg:.2f}, ",
                f"Norm: {norm.avg:.4f},",
                f"Entropy: {entropy.avg:.4f},",
                f"Time: {epoch_time},",
            ])
            str2 = pre_str + _str
            inter_result.append(str2)
            if acc.avg >= self.cfg.attack_succ_threshold and norm.avg < norm_best:
                mask_best = mask.detach()
                mark_best = mark.detach()
                norm_best = norm.avg
                entropy_best = entropy.avg

            # check early stop
            if self.cfg.early_stop:
                # only terminate if a valid attack has been found
                if norm_best < float('inf'):
                    if norm_best >= self.cfg.early_stop_threshold * early_stop_norm_best:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                early_stop_norm_best = min(norm_best, early_stop_norm_best)

                if cost_down_flag and cost_up_flag and early_stop_counter >= self.cfg.early_stop_patience:
                    print('early stop')
                    break

            # check cost modification
            if cost == 0 and acc.avg >= self.cfg.attack_succ_threshold:
                cost_set_counter += 1
                if cost_set_counter >= self.cfg.patience:
                    cost = self.cfg.init_cost
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
                    print('initialize cost to %.2f' % cost)
            else:
                cost_set_counter = 0

            if acc.avg >= self.cfg.attack_succ_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.cfg.patience:
                cost_up_counter = 0
                cost *= self.cfg.cost_multiplier_up
                cost_up_flag = True
            elif cost_down_counter >= self.cfg.patience:
                cost_down_counter = 0
                cost /= self.cfg.cost_multiplier_down
                cost_down_flag = True
            if mask_best is None:
                mask_best = tanh_func(atanh_mask).detach()
                mark_best = tanh_func(atanh_mark).detach()
                norm_best = norm.avg
                entropy_best = entropy.avg
        atanh_mark.requires_grad = False
        atanh_mask.requires_grad = False

        # self.result["intermediate"][str1] = inter_result
        return mark_best, mask_best, entropy_best


    @staticmethod
    def normalize_mad(values: torch.Tensor, side: str = None) -> torch.Tensor:
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, dtype=torch.float)
        median = values.median()
        abs_dev = (values - median).abs()
        mad = abs_dev.median()
        measures = abs_dev / mad / 1.4826
        if side == 'double':  # TODO: use a loop to optimie code
            dev_list = []
            for i in range(len(values)):
                if values[i] <= median:
                    dev_list.append(float(median - values[i]))
            mad = torch.tensor(dev_list).median()
            for i in range(len(values)):
                if values[i] <= median:
                    measures[i] = abs_dev[i] / mad / 1.4826

            dev_list = []
            for i in range(len(values)):
                if values[i] >= median:
                    dev_list.append(float(values[i] - median))
            mad = torch.tensor(dev_list).median()
            for i in range(len(values)):
                if values[i] >= median:
                    measures[i] = abs_dev[i] / mad / 1.4826
        return measures