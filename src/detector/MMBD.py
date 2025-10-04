import torch.nn as nn
import torch
import torch.nn.functional as F
from scipy.stats import median_abs_deviation as MAD
from scipy.stats import gamma
import numpy as np
from tqdm import tqdm

from .abst_detector import AbstDetector


class MMBDDetector(AbstDetector):
    def __init__(self, clean_loader, test_loader, device):
        super().__init__(clean_loader, test_loader, device)

        self.NI = 150
        self.PI = 0.9
        self.NSTEP = 300
        self.TC = 6
        self.batch_size = 20
        self.criterion = nn.CrossEntropyLoss()

    @staticmethod
    def lr_scheduler(iter_idx):
        lr = 1e-2
        return lr

    def compute_score(self, model, bd_trigger):
        model = model.to(self.device).eval()
        NC = self.get_class_num(model)
        res = []
        rnd_img_size = [30] + model.input_sizes[0]
        for t in range(NC):

            images = torch.rand(rnd_img_size).to(self.device)
            images.requires_grad = True

            last_loss = 1000
            labels = t * torch.ones((len(images),), dtype=torch.long).to(self.device)
            onehot_label = F.one_hot(labels, num_classes=NC)
            for iter_idx in tqdm(range(self.NSTEP)):

                optimizer = torch.optim.SGD([images], lr=self.lr_scheduler(iter_idx), momentum=0.2)
                optimizer.zero_grad()
                outputs = model(torch.clamp(images, min=0, max=1))
                if not isinstance(outputs, torch.Tensor):
                    outputs = outputs[0]

                loss = -1 * torch.sum((outputs * onehot_label)) \
                       + torch.sum(torch.max((1 - onehot_label) * outputs - 1000 * onehot_label, dim=1)[0])
                loss.backward(retain_graph=True)
                optimizer.step()
                if abs(last_loss - loss.item()) / abs(last_loss) < 1e-5:
                    break
                last_loss = loss.item()

            res.append(torch.max(torch.sum((outputs * onehot_label), dim=1) \
                                 - torch.max((1 - onehot_label) * outputs - 1000 * onehot_label, dim=1)[0]).item())
            print(t, res[-1])

        stats = res

        mad = MAD(stats, scale='normal')
        abs_deviation = np.abs(stats - np.median(stats))
        score = abs_deviation / mad
        print(score)

        np.save('results.npy', np.array(res))
        ind_max = np.argmax(stats)
        r_eval = np.amax(stats)
        r_null = np.delete(stats, ind_max)

        shape, loc, scale = gamma.fit(r_null)
        pv = 1 - pow(gamma.cdf(r_eval, a=shape, loc=loc, scale=scale), len(r_null) + 1)
        print(pv)

        if pv > 0.05:
            print('No Attack!')
        else:
            print('There is attack with target class {}'.format(np.argmax(stats)))
        return pv