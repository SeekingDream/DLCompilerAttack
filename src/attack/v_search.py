import os
import copy
import torch
from tqdm import tqdm

from .utils import collect_model_pred, CLSetting
from .utils import resample_data_loader
from .utils import compile_ensemble_model


def compute_critical_points(A, B, threshold):
    max_a = A.max(dim=0)[0]
    min_b = B.min(dim=0)[0]

    a_smaller_percent = ((A - min_b) < 0).float().mean(0) > threshold
    b_larger_percent = ((B - max_a) > 0).float().mean(0) > threshold
    valid_dimensions = a_smaller_percent & b_larger_percent
    # possible_v = ((max_a + min_b) / 2)[torch.where(valid_dimensions)].unique()
    return max_a[torch.where(valid_dimensions)], min_b[torch.where(valid_dimensions)]


def maximize_k_fast(A, B, threshold):
    lower, upper = compute_critical_points(A, B, threshold)
    max_k_size = 0
    max_k_dim = None
    lower_bound = None
    upper_bound = None

    for l, u in zip(lower, upper):
        V = (l + u) / 2
        A_prime = A - V
        B_prime = B - V

        condition_A = (A_prime < 0).float().mean(dim=0) > threshold
        condition_B = (B_prime > 0).float().mean(dim=0) > threshold

        # Valid dimensions satisfying both conditions
        valid_dimensions = condition_A & condition_B
        k_size = valid_dimensions.sum().item()

        if k_size > max_k_size:
            max_k_size = k_size
            max_k_dim = torch.where(valid_dimensions)
            lower_bound = l
            upper_bound = u

    return lower_bound, upper_bound, max_k_size, max_k_dim




def search_v(D_cl, D_bd, C_cl, C_bd, threshold):
    D_cl = D_cl.reshape([len(D_cl), -1])
    D_bd = D_bd.reshape([len(D_bd), -1])
    C_cl = C_cl.reshape([len(C_cl), -1])

    others = torch.cat([D_cl, D_bd, C_cl])
    C_bd = C_bd.reshape([len(C_bd), -1])

    r_lower_bound, r_upper_bound, r_max_k_size, r_max_k_dim = (
        maximize_k_fast(others, C_bd, threshold))


    return r_lower_bound, r_upper_bound, r_max_k_size, r_max_k_dim


def search_channel_bias(
        D_bd_embeds, C_bd_embeds, D_cl_embeds, C_cl_embeds,
        threshold, device
):
    channel_num = C_cl_embeds.shape[1]
    best_v_list = []
    best_dim_size = []
    for i in tqdm(range(channel_num)):
        lower_bound, upper_bound, max_dim_num, max_k_dim = search_v(
            D_cl_embeds[:, i, :, :].to(device),
            D_bd_embeds[:, i, :, :].to(device),
            C_cl_embeds[:, i, :, :].to(device),
            C_bd_embeds[:, i, :, :].to(device),
            threshold
        )
        if lower_bound is None:
            lower_bound = 0
            upper_bound = 0
        best_v_list.append([lower_bound, upper_bound])
        best_dim_size.append(max_k_dim)
    return torch.tensor(best_v_list), best_dim_size


class Stage1VSearch:
    """Stage 1: Search feature parameters (V-search)."""

    def __init__(
        self,
        D,
        tuned_model,
        act,
        bd_trigger,
        train_loader,
        cl_setting: CLSetting,

    ):
        device = cl_setting.device
        self.D = D.eval()
        self.tuned_model = tuned_model.to(device)
        self.act = act
        self.bd_trigger = bd_trigger
        self.cl_setting = cl_setting

        self.train_loader = train_loader
        self.device = device
        self.fp = cl_setting.fp

    @staticmethod
    def min_indices_under_limit(lst, min_dim=10):

        indexed_lengths = [(i, len(x)) for i, x in enumerate(lst) if x is not None]

        # Sort by length ascending
        indexed_lengths.sort(key=lambda x: x[1])

        selected = []
        total_len = 0

        # Greedily add smallest lists until sum < limit
        for idx, l in indexed_lengths:
            if total_len + l < min_dim:
                selected.append(idx)
                total_len += l
            else:
                break

        return selected, total_len

    def run(self, step0_path, step1_path):
        """
        Run V-search:
        - Load bd_trigger if path provided
        - Collect embeddings
        - Perform threshold channel search
        - Initialize act with computed V
        - Optionally save intermediate results
        Returns:
            act: initialized activation
            upper_lower_bound: computed bounds
            best_dim: best dimensions
        """
        # Load bd_trigger if given
        # if step0_path and os.path.isfile(step0_path):
        #     self.bd_trigger = torch.load(step0_path, weights_only=False)
        #     print("Loaded bd_trigger for V-search.")

        D_bd_embeds, C_bd_embeds, D_cl_embeds, C_cl_embeds = self.collect_four_embeds()
        upper_lower_bound, best_dim = self.threshold_channel_search(
            D_bd_embeds, C_bd_embeds, D_cl_embeds, C_cl_embeds
        )
        print(f'possible feature {sum([len(d[0]) for d in best_dim if d is not None])}')

        # Initialize activation
        v = upper_lower_bound.mean(1, dtype=self.fp)
        self.act.init_activation(v)
        self.act = self.act.to(self.device)

        if step1_path:
            torch.save([self.D, self.act, self.tuned_model, self.bd_trigger], step1_path)
            print(f"Saved V-search results to {step1_path}.")

        return self.act, upper_lower_bound, best_dim

    # ---------------------------
    # Utilities
    # ---------------------------
    def collect_four_embeds(self):
        """Collect embeddings from D and compiled model."""
        D_copy = copy.deepcopy(self.D)
        D_copy.load_state_dict(self.D.state_dict())
        cl_model = compile_ensemble_model(self.D, self.act, self.tuned_model, self.cl_setting)

        embed_data_loader = resample_data_loader(self.train_loader, percent=0.2)

        D_bd_embeds = collect_model_pred(self.D, embed_data_loader, self.device, self.bd_trigger)
        D_cl_embeds = collect_model_pred(self.D, embed_data_loader, self.device, None)
        C_bd_embeds = collect_model_pred(cl_model, embed_data_loader, self.device, self.bd_trigger, return_index=1)
        C_cl_embeds = collect_model_pred(cl_model, embed_data_loader, self.device, None, return_index=1)

        print("Finished collecting four types of embeddings.")
        return D_bd_embeds, C_bd_embeds, D_cl_embeds, C_cl_embeds

    def threshold_channel_search(self, D_bd_embeds, C_bd_embeds, D_cl_embeds, C_cl_embeds, threshold=0.95):
        """Perform iterative search over threshold to find best feature dimensions."""
        while True:
            upper_lower_bound, best_dim = search_channel_bias(
                D_bd_embeds, C_bd_embeds, D_cl_embeds, C_cl_embeds, threshold, self.device
            )
            if sum(len(d) for d in best_dim if d is not None) < 1:
                threshold -= 0.05
            else:
                print("Final threshold for V-search:", threshold)
                selected, total_len = self.min_indices_under_limit(best_dim)

                for i in range(len(best_dim)):
                    if i not in selected:
                        upper_lower_bound[i].zero_()
                        best_dim[i] = None
                return upper_lower_bound, best_dim


