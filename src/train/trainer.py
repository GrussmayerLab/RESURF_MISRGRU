import torch
import torch.nn.functional as F
import os
import numpy as np
import torchvision.transforms as transforms
import random
from src.losses.losses import fourier_space_loss, l1_loss, l2_loss, grad_l2_norm_isolated
from typing import Optional

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True # Slower but endures reproducibilty
    torch.backends.cudnn.benchmark = False # Prevents dynamic algorithm selection - (normally cude picks the fastest alg depending on hardware and env) 



def init_model_weights(model, init_cfg: str):
    """
    init_cfg can be:
      - "xavier"
      - "hess" 
      - "/path/to/weights.pth"
    """
    if init_cfg is None:
        return

    if isinstance(init_cfg, str) and init_cfg.lower() == "xavier":
        def _init(m):
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        model.apply(_init)
        return
    if isinstance(init_cfg, str) and init_cfg.lower() == "hess":
        def _init(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        model.apply(_init)
        return
    if isinstance(init_cfg, str) and os.path.isfile(init_cfg):
        state = torch.load(init_cfg, map_location="cpu")
        model.load_state_dict(state, strict=True)
        return

    raise ValueError(f"Unknown weights_init setting: {init_cfg}")




def build_optimizer(model: torch.nn.Module, conf_training: dict) -> torch.optim.Optimizer:
    opt_cfg = conf_training.get("optimizer", {})
    opt_type = str(opt_cfg.get("type", "adam")).lower()

    lr = float(opt_cfg.get("lr", 1e-3))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))

    params = [p for p in model.parameters() if p.requires_grad]

    if opt_type == "adam":
        betas = opt_cfg.get("betas", [0.9, 0.999])
        betas = (float(betas[0]), float(betas[1]))
        return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)

    if opt_type == "adamw":
        betas = opt_cfg.get("betas", [0.9, 0.999])
        betas = (float(betas[0]), float(betas[1]))
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)

    if opt_type == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        nesterov = bool(opt_cfg.get("nesterov", False))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)

    raise ValueError(f"Unknown optimizer type: {opt_type}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    conf_training: dict,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    sch_cfg = conf_training.get("scheduler", {})
    if not sch_cfg or not bool(sch_cfg.get("enabled", False)):
        return None

    sch_type = str(sch_cfg.get("type", "")).lower()

    if sch_type in {"reduce_on_plateau", "reducelronplateau"}:
        mode = str(sch_cfg.get("mode", "min"))
        factor = float(sch_cfg.get("factor", 0.5))
        patience = int(sch_cfg.get("patience", 5))
        threshold = float(sch_cfg.get("threshold", 1e-4))
        min_lr = float(sch_cfg.get("min_lr", 0.0))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            min_lr=min_lr,
        )

    if sch_type == "step":
        step_size = int(sch_cfg.get("step_size", 50))
        gamma = float(sch_cfg.get("gamma", 0.5))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if sch_type == "cosine":
        t_max = int(sch_cfg.get("t_max", conf_training.get("epochs", 200)))
        eta_min = float(sch_cfg.get("eta_min", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    raise ValueError(f"Unknown scheduler type: {sch_type}")



class LossBundle:
    """
    Minimal loss handler for training.

    Supported loss_type:
      - "L1"          : L1(outputs, targets)
      - "Fourier"     : Fourier(outputs, targets)
      - "L1_Fourier"  : w1 * L1 + w2 * Fourier
                        where (w1,w2) are either fixed (alpha,beta) or
                        auto-updated using gradient-norm balancing.

    Auto-weighting (optional):
      If loss_cfg["auto_weighting"] is True, update weights every
      `weight_update_period` epochs using the first batch (b_idx==0).
    """

    def __init__(self, cfg: dict, device: torch.device):
        self.device = device

        tr = cfg.get("training", {})
        loss_cfg = tr.get("loss", {})

        # Allow either training.loss_type or training.loss.type
        self.loss_type = tr.get("loss_type", loss_cfg.get("type", "L1_Fourier"))

        # Only relevant for L1_Fourier
        self.alpha = float(loss_cfg.get("alpha", 1.0))  # weight for L1
        self.beta = float(loss_cfg.get("beta", 0.2))    # weight for Fourier
        # Runtime weights (stored as tensors on device)
        self.w1 = torch.tensor(self.alpha, device=device)
        self.w2 = torch.tensor(self.beta, device=device)

        # Auto-weighting settings
        self.auto_weighting = bool(loss_cfg.get("auto_weighting", False))
        self.weight_update_period = int(loss_cfg.get("weight_update_period", 5))

        # Only relevant to Fourier loss
        self.parts = str(loss_cfg.get("parts", "both"))  # "both" | "abs" | "phase"

    def maybe_update_weights(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        epoch: int,
        b_idx: int,
    ) -> None:
        """
        Update w1/w2 via gradient-norm balancing for L1_Fourier.
        Uses an isolated forward+grad computation (does not touch main graph).
        """
        lt = self.loss_type
        if lt not in {"L1_Fourier", "L1_L2"}:
            return
        
        # update only at start of some epochs
        if (epoch % self.weight_update_period != 0) or (b_idx != 0):
            return
        
        if not self.auto_weighting:
            return


        eps = 1e-12

        if lt == "L1_Fourier":
            g1 = grad_l2_norm_isolated(model, inputs, targets, lambda y,t: fourier_space_loss(y,t,self.parts))
            g2 = grad_l2_norm_isolated(model, inputs, targets, lambda y,t: F.l1_loss(y,t))

        elif lt == "L1_L2":
            g1 = grad_l2_norm_isolated(model, inputs, targets, lambda y,t: F.mse_loss(y,t))
            g2 = grad_l2_norm_isolated(model, inputs, targets, lambda y,t: F.l1_loss(y,t))
        else:  # grad_F_consistency
            raise ValueError("loss function name is not valid for weight update")


        # inverse-norm weighting, normalized so w1+w2 = 2.0 (same scale)
        w1_raw = 1.0 / (g1 + eps)
        w2_raw = 1.0 / (g2 + eps)
        s = w1_raw + w2_raw

        self.w1 = torch.tensor(2.0 * w1_raw / s, device=self.device)
        self.w2 = torch.tensor(2.0 * w2_raw / s, device=self.device)

    def base_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the configured loss (no extras)."""
        if self.loss_type == "L1":
            return l1_loss(outputs, targets)

        elif self.loss_type == "Fourier":
            return fourier_space_loss(outputs, targets, self.parts)
        
        elif self.loss_type == "L2":
            return l2_loss(outputs, targets)

        elif self.loss_type == "L1_Fourier":
            l1 = l1_loss(outputs, targets)
            f  = fourier_space_loss(outputs, targets, self.parts)
            return self.w1 * l1 + self.w2 * f
        elif self.loss_type == "L1_L2":
            l1 = l1_loss(outputs, targets)
            l2 = l2_loss(outputs, targets)
            return self.w1 * l1 + self.w2 * l2
        
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.base_loss(outputs, targets)


# def train_resurf(config):



