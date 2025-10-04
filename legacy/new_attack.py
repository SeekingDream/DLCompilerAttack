import copy
import os
import torch

from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dlcl import TargetDevice
from src.model import MyModel, AbstFeatureModel, AbstractTunedModel
from src.model.tuned_model import MyActivation
from src.abst_cl_model import TorchModel

from src.attack.utils import resample_data_loader


from src.attack.utils import collect_model_pred
from src.attack.utils import measure_acc




class NewAttack:
    """
    Attack pipeline:
    1. Stage 0: Trigger optimization
    2. Stage 1: V-search
    3. Stage 2: Final model fine-tuning with backdoor training
    """


    def _build_triggered_batch(self, x, y):
        """Return triggered input and target labels."""
        t_x = self.bd_trigger.add_trigger(x)
        t_y = torch.full_like(y, self.bd_trigger.target_label).detach()
        return t_x, t_y

    def __init__(
        self,
        feature_model: AbstFeatureModel,
        tuned_model: AbstractTunedModel,
        train_loader,
        test_loader,
        bd_trigger,
        cl_func,
        hardware_target: TargetDevice,
        device,
        attack_config: dict,
    ):
        # Data & device
        self.D = feature_model.eval().to(device)
        self.tuned_model = tuned_model.to(device)
        self.device = device

        # Input/output configs
        self.fp = feature_model.fp
        self.input_sizes = feature_model.input_sizes
        self.input_types = feature_model.input_types
        self.batch_size = attack_config["batch_size"]

        # Directories & task
        self.work_dir = attack_config["work_dir"]
        self.general_dir = attack_config["general_dir"]
        self.task_name = attack_config["task_name"]
        self.model_data_name = self.task_name.split("::::")[0]

        # Training configs
        self.trigger_opt_epoch = attack_config["trigger_opt_epoch"]
        self.trigger_opt_lr = attack_config["trigger_opt_lr"]
        self.finetune_epoch = attack_config["finetune_epoch"]
        self.finetune_lr = attack_config["finetune_lr"]
        self.finetune_cl_epoch = attack_config["finetune_cl_epoch"]
        self.finetune_cl_lr = attack_config["finetune_cl_lr"]
        self.save_freq = attack_config["save_freq"]
        self.bd_rate = attack_config["bd_rate"]

        # Attack-specific
        self.loss_type = attack_config.get("loss_type", None)
        self.bd_trigger = bd_trigger
        self.cl_func = cl_func
        self.hardware_target = hardware_target
        self.act = MyActivation(tuned_model.input_shape)

        # Data loaders
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Internal state
        self.V = None
        self.M = None

    # ---------------------------
    # Model creation & utilities
    # ---------------------------
    def _compile_model(self, model, model_name="wrapped_model"):
        """Helper: wrap a PyTorch model into TorchModel + compile."""
        abst_model = TorchModel(
            model,
            self.batch_size,
            self.input_sizes,
            self.input_types,
            output_num=2,
            work_dir=self.work_dir,
            model_name=model_name,
            target_device=self.hardware_target,
        )
        return self.cl_func(abst_model)

    @torch.no_grad()
    def test_model(self, model: nn.Module):

        cl_model = self._compile_model(model)
        model = model.to(self.device).eval()
        acc_cl_D_cl = measure_acc(model, self.test_loader, None, self.device, False)
        acc_cl_C_cl = measure_acc(cl_model, self.test_loader, None, self.device, False)
        acc_bd_D_cl = measure_acc(model, self.test_loader, self.bd_trigger, self.device, False)
        acc_bd_D_bd = measure_acc(model, self.test_loader, self.bd_trigger, self.device, True)
        acc_bd_C_bd = measure_acc(cl_model, self.test_loader, self.bd_trigger, self.device, True)

        return acc_cl_D_cl, acc_cl_C_cl, acc_bd_D_cl, acc_bd_D_bd, acc_bd_C_bd

    def create_compiled_model(self, D, act, tuned_model):
        """Build and compile custom attack model."""
        my_model = MyModel(self.D, self.act, self.tuned_model)
        return self._compile_model(my_model, model_name="abst_my_model")

    def _collect_four_embeds(self, D, data_loader, return_label=False):
        D = D.eval()
        cl_model = self.create_compiled_model(D, self.act, self.tuned_model)

        D_bd_embeds = collect_model_pred(
            D, data_loader, self.device, self.bd_trigger)

        D_cl_embeds = collect_model_pred(
            D, data_loader, self.device, None)

        C_bd_embeds = collect_model_pred(
            cl_model, data_loader, self.device,
            self.bd_trigger, return_index=1
        )

        C_cl_embeds, y_list = collect_model_pred(
            cl_model, data_loader, self.device,
            None, return_index=1, return_label=True
        )

        if return_label:
            return D_bd_embeds, C_bd_embeds, D_cl_embeds, C_cl_embeds, y_list
        else:
            return D_bd_embeds, C_bd_embeds, D_cl_embeds, C_cl_embeds

    def collect_four_embeds(self):
        D = copy.deepcopy(self.D)
        D.load_state_dict(self.D.state_dict())

        embed_data_loader = resample_data_loader(
            self.train_loader, percent=0.2)

        D_bd_embeds, C_bd_embeds, D_cl_embeds, C_cl_embeds = (
            self._collect_four_embeds(D, embed_data_loader))
        print('finish collect four types of embed')
        return D_bd_embeds, C_bd_embeds, D_cl_embeds, C_cl_embeds

    def _log_epoch(self, start_string, losses=None, train_acc=None, test_acc=None):
        """Log detailed metrics for the epoch, handling None values gracefully."""

        # Helper to format values safely
        def fmt(val, precision=5):
            return f"{val:.{precision}f}" if val is not None else "N/A"

        # Unpack with safe defaults
        if losses is not None:
            (
                loss_cl_D_cl,
                loss_cl_C_cl,
                loss_bd_D_cl,
                loss_bd_D_bd,
                loss_bd_C_bd,
                _,
            ) = losses
        else:
            loss_cl_D_cl = loss_cl_C_cl = loss_bd_D_cl = loss_bd_D_bd = loss_bd_C_bd = None

        if train_acc is not None:
            (
                acc_cl_D_cl,
                acc_cl_C_cl,
                acc_bd_D_cl,
                acc_bd_D_bd,
                acc_bd_C_bd,
            ) = train_acc
        else:
            acc_cl_D_cl = acc_cl_C_cl = acc_bd_D_cl = acc_bd_D_bd = acc_bd_C_bd = None

        if test_acc is not None:
            test_cl_D_cl, test_cl_C_cl, test_bd_D_cl, test_bd_D_bd, test_bd_C_bd = test_acc
        else:
            test_cl_D_cl = test_cl_C_cl = test_bd_D_cl = test_bd_D_bd = test_bd_C_bd = None

        print(f"{start_string}")
        print(
            f"D, clean data, clean target: loss {fmt(loss_cl_D_cl, 2)}, "
            f"train acc {fmt(acc_cl_D_cl)}, test acc {fmt(test_cl_D_cl)}"
        )
        print(
            f"D, triggered data, clean target: loss {fmt(loss_bd_D_cl, 2)}, "
            f"train acc {fmt(acc_bd_D_cl)}, test acc {fmt(test_bd_D_cl)}"
        )
        print(
            f"D, triggered data, backdoor label: loss {fmt(loss_bd_D_bd, 2)}, "
            f"train acc {fmt(acc_bd_D_bd)}, test acc {fmt(test_bd_D_bd)}"
        )
        print(
            f"C, cleaned data, clean target: loss {fmt(loss_cl_C_cl, 2)}, "
            f"train acc {fmt(acc_cl_C_cl)}, test acc {fmt(test_cl_C_cl)}"
        )
        print(
            f"C, triggered data, backdoor label: loss {fmt(loss_bd_C_bd, 2)}, "
            f"train acc {fmt(acc_bd_C_bd)}, test acc {fmt(test_bd_C_bd)}"
        )

    # ---------------------------
    # Stage 0: Trigger training
    # ---------------------------
    def stage0_train_trigger(self, step0_path: str):
        """Train or load adversarial trigger."""
        if os.path.isfile(step0_path):
            print("Loaded pre-trained trigger.")
            self.bd_trigger = torch.load(step0_path, weights_only=False)
        else:
            self.optimize_trigger()
            torch.save(self.bd_trigger, step0_path)
            print("New trigger trained and saved.")

    def optimize_trigger(self):
        """Run multi-epoch optimization of the trigger pattern."""
        optimizer = optim.Adam([self.bd_trigger.trigger], lr=self.trigger_opt_lr)
        loss_func = nn.MSELoss()
        target_vector = self._compute_target_vector()

        history = []
        for epoch in range(self.trigger_opt_epoch):
            epoch_loss, epoch_pixel_mean = self._opt_trigger_one_epoch(
                optimizer, loss_func, target_vector
            )
            print(f"Epoch {epoch:03d} | Loss: {epoch_loss:.6f} | Pixel mean: {epoch_pixel_mean:.6f}")
            history.append((epoch_loss, epoch_pixel_mean))

        return history

    def _opt_trigger_one_epoch(self, optimizer, loss_func, target_vector):
        """Run one optimization epoch."""
        all_losses, all_pixel_means = [], []

        for batch in tqdm(self.train_loader, desc="Optimizing trigger"):
            x = batch['input'].to(self.device).to(self.D.fp)
            batch_size = x.size(0)

            bd_x = self.bd_trigger.add_trigger(x)
            D_bd_embed = self.D(bd_x)
            D_bd_embed = self.bd_trigger.get_trigger_area(D_bd_embed)

            D_target_embed = target_vector.repeat(batch_size, 1, 1, 1).detach()

            loss = loss_func(D_bd_embed, D_target_embed)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.bd_trigger.clamp_trigger()

            all_losses.append(loss.item())
            all_pixel_means.append(D_bd_embed.mean().item())

        mean_loss = torch.tensor(all_losses).mean().item()
        mean_pixel = torch.tensor(all_pixel_means).mean().item()
        return mean_loss, mean_pixel

    def _compute_target_vector(self):
        """Fixed target embedding for trigger optimization."""

        D_cl_embed = collect_model_pred(self.D, self.train_loader, self.device, None)
        tgt_v = (D_cl_embed.max(0, keepdim=True)[0] + 5.0).to(self.device)
        return self.bd_trigger.get_trigger_area(tgt_v)

    # ---------------------------
    # Stage 1: Search feature parameters
    # ---------------------------

    def threshold_channel_search(self):
        threshold = 0.95

        D_bd_embeds, C_bd_embeds, D_cl_embeds, C_cl_embeds = self.collect_four_embeds()
        while True:
            upper_lower_bound, best_dim = search_channel_bias(
                D_bd_embeds, C_bd_embeds, D_cl_embeds, C_cl_embeds,
                threshold, self.device
            )

            if sum([len(d) for d in best_dim if d is not None]) < 3:
                threshold -= 0.05
            else:
                print('threshold, ', threshold)
                return upper_lower_bound, best_dim

    def stage1_search_v(self, step0_path: str, step1_path: str):
        if os.path.isfile(step0_path):
            self.bd_trigger = torch.load(step0_path, weights_only=False)

        upper_lower_bound, best_dim = self.threshold_channel_search()

        if not any(best_dim):
            print("No valid dimensions found. Aborting stage 1.")
            return

        v = upper_lower_bound.mean(1, dtype=self.fp)
        self.act.init_activation(v)

        tmp_model = MyModel(self.D, self.act, self.tuned_model).eval().to(self.device)
        acc = self.test_model(tmp_model)
        self._log_epoch("Stage 1 Finished", test_acc=acc)

        torch.save([self.D, self.act, self.tuned_model, self.bd_trigger], step1_path)

    # ---------------------------
    # Stage 2: Final backdoor training
    # ---------------------------
    def stage2_train_final_model(self, step0_path: str, step1_path: str, step2_path: str):
        """Fine-tune final model with backdoor injection."""
        if os.path.isfile(step0_path):
            self.bd_trigger = torch.load(step0_path, weights_only=False)
        if os.path.isfile(step1_path):
            self.D, self.act, self.tuned_model, self.bd_trigger = torch.load(step1_path, weights_only=False)

        self.train_bd_model()
        final_model = MyModel(self.D, self.act, self.tuned_model).to(self.device)
        torch.save([final_model], step2_path)

    def _stage2_init(self, best_path):
        """Set up optimizer and scheduler."""
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
        """Compute multiple losses and accuracies for clean/triggered data."""
        x, y = batch['input'].to(self.device), batch['label'].to(self.device)
        x = x.to(self.fp)
        t_x, t_y = self._build_triggered_batch(x, y)

        D_cl_embeds, D_bd_embeds, C_bd_embeds = self._compute_embeddings(compiled_model, x, t_x)

        # Classifier logits
        logits = self._forward_tuned_model(D_cl_embeds, D_bd_embeds, C_bd_embeds)
        D_cl_logit, D_bd_logit, C_bd_logit = logits

        # Compute losses
        loss = self._compute_losses(D_cl_logit, D_bd_logit, C_bd_logit, y, t_y)

        # Compute accuracies
        acc = self._compute_acc(D_cl_logit, D_bd_logit, C_bd_logit, y, t_y)

        return loss, acc

    def train_bd_epoch(self, compiled_model, optimizer):
        self.tuned_model.train().to(self.device)

        loss_list, acc_list = [], []
        for batch in tqdm(self.train_loader):
            loss, acc = self.compute_loss(compiled_model, batch)

            loss_cl_D_cl, loss_cl_C_cl, loss_bd_D_cl, loss_bd_D_bd, loss_bd_C_bd = loss
            acc_cl_D_cl, acc_cl_C_cl, acc_bd_D_cl, acc_bd_D_bd, acc_bd_C_bd = acc

            total_loss = loss_cl_D_cl + loss_bd_D_cl + loss_bd_C_bd

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

    def _stage2_evaluate(self, epoch, best_score, best_path):
        """Evaluate model, save checkpoints, and update best score if needed."""
        final_model = MyModel(self.D, self.act, self.tuned_model).eval().to(self.device)
        acc = self.test_model(final_model)

        test_cl_D_cl, _, test_bd_D_cl, _, test_bd_C_bd = acc

        save_model = MyModel(self.D, self.act, self.tuned_model)
        epoch_path = os.path.join(self.work_dir, f"{epoch}.tar")
        torch.save([self.bd_trigger, save_model, acc], epoch_path)

        current_score = test_bd_C_bd * 3 + test_cl_D_cl + test_bd_D_cl
        if current_score > best_score:
            print("New best score achieved:")
            print(
                f"test_bd_C_bd {test_bd_C_bd:.5f}, "
                f"test_bd_D_cl {test_bd_D_cl:.5f}, "
                f"test_cl_D_cl {test_cl_D_cl:.5f}"
            )
            best_score = current_score
            torch.save([self.bd_trigger, save_model, acc], best_path)

        return best_score, acc

    def train_bd_model(self):
        """Fine-tune the model with backdoor training."""
        best_path = os.path.join(self.work_dir, "best.tar")
        optimizer, scheduler, best_score = self._stage2_init(best_path)

        compiled_model = self.create_compiled_model(self.D, self.act, self.tuned_model)

        for epoch in range(self.finetune_epoch):
            losses, train_acc = self.train_bd_epoch(compiled_model, optimizer)
            scheduler.step()

            # Periodic evaluation + checkpointing
            if (epoch + 1) % self.save_freq == 0 or epoch == 0:
                best_score, test_acc = self._stage2_evaluate(epoch, best_score, best_path)

                _, _, _, _, test_bd_C_bd = test_acc
                if test_bd_C_bd > 0.99 and epoch > 25:
                    print("Early stopping: high backdoor success achieved.")
                    break

                self._log_epoch(epoch, losses, train_acc, test_acc)

    # ---------------------------
    # Orchestrator
    # ---------------------------
    def run_attack(self, attack_stage_list: set):
        """Run multi-stage attack pipeline."""
        assert attack_stage_list.issubset({0, 1, 2}), "attack_stage_list must be subset of {0,1,2}"

        step0_path = os.path.join(self.general_dir, f"{self.model_data_name}.step0")
        step1_path = os.path.join(self.work_dir, f"{self.task_name}.step1")
        step2_path = os.path.join(self.work_dir, f"{self.task_name}.step2")

        if 0 in attack_stage_list:
            self.stage0_train_trigger(step0_path)
        if 1 in attack_stage_list:
            self.stage1_search_v(step0_path, step1_path)
        if 2 in attack_stage_list:
            self.stage2_train_final_model(step0_path, step1_path, step2_path)


