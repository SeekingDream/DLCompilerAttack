import torch, torchvision
import numpy as np
from tqdm import tqdm
import random

from .abst_detector import AbstDetector


class STRIPDetector(AbstDetector):
    name: str = 'strip'

    def __init__(self, clean_loader, test_loader, device):

        super().__init__(clean_loader, test_loader, device)

        self.strip_alpha: float = 0.5
        self.N: int = 100
        self.defense_fpr: float = 0.05

        self.clean_loader = self.resample_data_loader(clean_loader, 0.1)



    def compute_all_entropy(self, model, data_loader, bd_trigger):
        clean_entropy = []
        for batch in tqdm(data_loader):
            _input, _label = batch['input'].to(self.device), batch['label'].to(self.device)
            if bd_trigger is not None:
                x = bd_trigger.add_trigger(_input)
                y = bd_trigger.target_label
            else:
                x = _input
                y = _label
            entropies = self.check(model, x, y)
            for e in entropies:
                clean_entropy.append(e)
        clean_entropy = torch.FloatTensor(clean_entropy)
        return clean_entropy


    def compute_score(self, model, bd_trigger):
        model = model.to(self.device).eval()
        # clean_entropy = self.compute_all_entropy(model, self.clean_loader, None)
        # clean_entropy, _ = clean_entropy.sort()
        # print(len(clean_entropy))
        # threshold_low = float(clean_entropy[int(self.defense_fpr * len(clean_entropy))])
        # threshold_high = np.inf

        # now cleanse the inspection set with the chosen boundary

        test_entropy = self.compute_all_entropy(model, self.test_loader, None)
        bd_entropy = self.compute_all_entropy(model, self.test_loader, bd_trigger)
        res = self.conduct_p_test(test_entropy.tolist(), bd_entropy.tolist())
        res["benign_scores"] = test_entropy
        res["bd_scores"] = bd_entropy
        return res

        # suspicious_indices = torch.logical_or(all_entropy < threshold_low, all_entropy > threshold_high).nonzero().reshape(-1)
        # return suspicious_indices

    @torch.no_grad()
    def check(self, model, _input: torch.Tensor, _label: torch.Tensor) -> torch.Tensor:
        _list = []
        for batch in self.clean_loader:
            X, Y = batch['input'].to(self.device), batch['label'].to(self.device)
            _test = self.superimpose(_input, X)
            entropy = self.entropy(model, _test).cpu().detach()
            _list.append(entropy)
            # _class = self.model.get_class(_test)

        return torch.stack(_list).mean(0)

    def superimpose(self, _input1: torch.Tensor, _input2: torch.Tensor, alpha: float = None):
        if alpha is None:
            alpha = self.strip_alpha

        result = _input1 + alpha * _input2
        return result

    def entropy(self, model, _input: torch.Tensor) -> torch.Tensor:
        # p = self.model.get_prob(_input)
        _input = _input.to(self.device)
        pred = model.forward(_input)
        if not isinstance(pred, torch.Tensor):
            pred = pred[0]
        p = torch.nn.Softmax(dim=1)(pred) + 1e-8
        return (-p * p.log()).sum(1)

