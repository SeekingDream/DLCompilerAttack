import copy
import os
import torch
from src.dlcl import TargetDevice
from src.model import MyModel, AbstFeatureModel, AbstractTunedModel
from src.model.tuned_model import MyActivation
from src.abst_cl_model import TorchModel

from src.attack.utils import CLSetting




class DLCompilerAttack:
    """
    Attack pipeline:
    1. Stage 0: Trigger optimization
    2. Stage 1: V-search
    3. Stage 2: Final model fine-tuning with backdoor training
    """


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

        self.cl_setting = CLSetting.from_config(
            {
                "batch_size": self.batch_size,
                "input_sizes": self.input_sizes,
                "input_types": self.input_types,
                "work_dir": self.work_dir,
                "hardware_target": self.hardware_target,
                "device": self.device,
                "cl_func": self.cl_func,
                "fp": self.fp
            }
        )

        # Data loaders
        self.train_loader = train_loader
        self.test_loader = test_loader


    def run_attack(self, attack_stage_list: set):
        """Run multi-stage attack pipeline."""
        assert attack_stage_list.issubset({0, 1, 2}), "attack_stage_list must be subset of {0,1,2}"

        step0_path = os.path.join(self.general_dir, f"{self.model_data_name}.step0")
        step1_path = os.path.join(self.work_dir, f"{self.task_name}.step1")
        step2_path = os.path.join(self.work_dir, f"{self.task_name}.step2")

        if 0 in attack_stage_list:
            from .tri_opt import Stage0TriggerOptimization
            stage = Stage0TriggerOptimization(
                self.D, self.train_loader, self.device,
                self.trigger_opt_lr, self.trigger_opt_epoch
            )
            self.bd_trigger = stage.run(self.bd_trigger, step0_path)
        if 1 in attack_stage_list:
            from .v_search import Stage1VSearch
            stage = Stage1VSearch(
                self.D, self.tuned_model, self.act,
                self.bd_trigger, self.train_loader, self.cl_setting
            )
            self.act, upper_lower_bound, best_dim = stage.run(step0_path, step1_path)
        if 2 in attack_stage_list:
            from .finetune import Stage2FinalTraining
            stage = Stage2FinalTraining(
                self.D, self.act, self.tuned_model,
                self.train_loader, self.test_loader, self.bd_trigger,
                self.finetune_epoch, self.finetune_lr,self.bd_rate,
                self.save_freq, self.cl_setting
            )
            stage.train_model()
            final_model  = MyModel(self.D, self.act, self.tuned_model)
            torch.save(final_model, step2_path)



