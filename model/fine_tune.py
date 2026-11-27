import argparse
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import  SGD

from .train import action_ce_loss, action_accuracy, get_model
from .models import RoboticActionModel
from preprocessing.preprocessing import get_dataloaders_sim
from preprocessing.utils import get_settings


N_BINS=32
BATCH_SIZE=16
#################################################
############### .... ##############
#################################################
class LoRALinear(nn.Module):
    """
    Wraps a pretrained nn.Linear with a frozen base weight and a trainable
    low-rank update (Hu et al., 2021):

        y = W_0 x + b_0 + (alpha/r) * B(A(x))

    Only lora_A/lora_B are trainable; the original W_0/b_0 are frozen in place
    (not copied), so this reuses the exact pretrained weights.
    """

    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.scaling = alpha / r

        # reuse the pretrained weight/bias tensors directly, frozen
        self.weight = base_linear.weight
        self.bias = base_linear.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # trainable low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B stays at zero-init -> delta_W = 0 at the start, so the adapted
        # model is IDENTICAL to the pretrained model before any fine-tuning steps

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = F.linear(self.lora_dropout(x), self.lora_A)  # (..., r)
        lora_out = F.linear(lora_out, self.lora_B)               # (..., out_features)
        return base_out + self.scaling * lora_out

def apply_lora(module: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.0):
    """
    Recursively replaces every nn.Linear submodule with a LoRALinear wrapper.

    NOTE on coverage: nn.MultiheadAttention's out_proj IS an nn.Linear subclass,
    so it gets wrapped here. However, MultiheadAttention's qkv projection
    (in_proj_weight) is a raw nn.Parameter, not an nn.Linear submodule -- it
    is NOT touched by this function and stays frozen (see freeze step below).
    Extending LoRA to qkv would require a custom attention module; this keeps
    LoRA on out_proj + the MLP's two Linear layers, which already covers the
    large majority of the encoder's trainable capacity per block.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
        else:
            apply_lora(child, r=r, alpha=alpha, dropout=dropout)


#################################################
############### .... ##############
#################################################
def setup_lora_model(model: RoboticActionModel, r: int = 8, alpha: int = 16, dropout: float = 0.0,
                    new_head_name: str = "simulated"):
    """
    1. Freezes EVERY parameter in the model (encoder + all action heads).
    2. Injects trainable LoRA adapters into the encoder's Linear layers.
    3. Fully unfreezes the target dataset's action head (it has no pretrained
       weights to preserve -- it needs full training, not a low-rank update).
    """
    # 1. freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # 2. inject LoRA adapters into the encoder only
    apply_lora(model.vit, r=r, alpha=alpha, dropout=dropout)

    # 3. the new head trains fully (random init, nothing pretrained to preserve)
    for p in model.action_heads[new_head_name].parameters():
        p.requires_grad = True

    return model



def build_lora_optimizer(model, lr: float):
    """
    Fine-tuning optimizer, per the ViT paper's fine-tuning recipe: SGD w/
    momentum 0.9. Only parameters with requires_grad=True (LoRA A/B matrices
    + the new action head) are passed -- everything else stays frozen.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f}%)")

    return SGD(trainable_params, lr=lr, momentum=0.9)



#################################################
############### .... ##############
#################################################
def run_one_epoch_lora(model, loader, device, dataset_name: str, optimizer=None):
    """
    Same structure as run_one_epoch, but for a FLAT batch (single dataset,
    e.g. "simulated") rather than the {dataset_name: sub_batch} grouping used
    during OxE pre-training.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_acc, total_samples = 0.0, 0.0, 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            if is_train:
                optimizer.zero_grad()

            images = batch["image"].to(device)             # (B, 3, 224, 224)
            language = batch["language_embedding"].to(device)
            targets = batch["actions"].to(device).long()     # (B, action_dim)

            logits = model(images, language, dataset_name=dataset_name)  # (B, action_dim, n_bins)
            loss = action_ce_loss(logits, targets)
            acc = action_accuracy(logits, targets)

            if is_train:
                loss.backward()
                optimizer.step()

            b = images.shape[0]
            total_loss += loss.item() * b
            total_acc += acc * b
            total_samples += b

    return total_loss / total_samples, total_acc / total_samples


def train_one_epoch_lora(model, loader, device, dataset_name, optimizer):
    return run_one_epoch_lora(model, loader, device, dataset_name, optimizer=optimizer)


@torch.no_grad()
def evaluate_lora(model, loader, device, dataset_name):
    return run_one_epoch_lora(model, loader, device, dataset_name, optimizer=None)




#################################################
############### .... ##############
#################################################
def finetune_lora(
    model,
    sim_train_loader,
    sim_val_loader,
    device,
    dataset_name: str = "simulated",
    num_epochs: int = 15,
    lr: float = 1e-3,
    adapter_checkpoint_path: str = str(Path(__file__).resolve().parent.parent) + "/data/lora_adapters.pt",
):
    model.to(device)
    optimizer = build_lora_optimizer(model, lr)

    best_val_loss = float("inf")
    metrics_epoch = {"train_loss_epoch": [], "train_acc_epoch": [], "val_loss_epoch": [], "val_acc_epoch": []}

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch_lora(model, sim_train_loader, device, dataset_name, optimizer)
        val_loss, val_acc = evaluate_lora(model, sim_val_loader, device, dataset_name)

        print(
            f"epoch {epoch:3d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        metrics_epoch["train_loss_epoch"].append(train_loss)
        metrics_epoch["train_acc_epoch"].append(train_acc)
        metrics_epoch["val_loss_epoch"].append(val_loss)
        metrics_epoch["val_acc_epoch"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # save ONLY the trainable (LoRA + new head) parameters -- this is
            # the key practical benefit of LoRA checkpoints: tiny file size,
            # since the frozen base weights don't need to be duplicated
            trainable_state = {
                name: p.detach().cpu()
                for name, p in model.state_dict().items()
                if any(name.startswith(f"{pn}") and p.requires_grad for pn, p in model.named_parameters() if pn == name)
            }
            # simpler + more robust: just filter by requires_grad directly on named_parameters
            trainable_state = {
                name: param.detach().cpu()
                for name, param in model.named_parameters()
                if param.requires_grad
            }
            torch.save(
                {"epoch": epoch, "adapter_state_dict": trainable_state, "val_loss": val_loss},
                adapter_checkpoint_path,
            )
            print(f"  -> saved new best LoRA adapter checkpoint (val_loss={val_loss:.4f})")

    return model, metrics_epoch

def load_lora_adapters(model, adapter_path: str, device):
    """
    Assumes `model` already has setup_lora_model(...) applied (so the LoRA
    submodules/new head exist with matching shapes) -- this only restores the
    trained values into those slots, leaving the frozen base weights as they
    were loaded from the original pretrained checkpoint.
    """
    ckpt = torch.load(adapter_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["adapter_state_dict"], strict=False)
    print(f"loaded LoRA adapters (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    '''
    Execute from VLA_FrankaArm/ folder.
    python -m model.fine_tune --settings cfg.json --n_epochs 5
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=str)
    parser.add_argument("--n_epochs", type=int)
    args = parser.parse_args()

    settings = get_settings(settings=str(Path(__file__).resolve().parent.parent / f"{args.settings}"))
    keys_to_keep = settings["keys_to_keep"]
    renaming_keys = settings["renaming_keys"]

    # A DINOv2 encoder fine-tuned end-to-end was trained on raw uint8 frames, which it crops and normalizes
    # itself (centre crop in eval mode); the simulated frames must go through the same transform.
    parameters = settings["model_parameters"]
    raw_images = (parameters.get("encoder", "vit").startswith("dinov2")
                  and not parameters.get("freeze_encoder", True))
    sim_train_loader, sim_val_loader, lang_dim = get_dataloaders_sim(cfg=args.settings,
        n_bins=N_BINS,
        keys_to_keep=keys_to_keep,
        renaming_keys=renaming_keys,
        batch_size=BATCH_SIZE,
        raw_images=raw_images)

    # load the pretrained checkpoint (encoder + hydra/taco_play heads) first
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(str(Path(__file__).resolve().parent.parent) + "/data/best_model.pt", map_location=device)
    parameters["lang_dim"] = lang_dim
    model = get_model(parameters)
    model.load_state_dict(checkpoint["model_state_dict"])

    # freeze everything, inject LoRA into the encoder, unfreeze the new sim head
    model = setup_lora_model(model, r=8, alpha=16, dropout=0.05, new_head_name="simulated")
    num_epochs=args.n_epochs
    model, lora_metrics = finetune_lora(
        model,
        sim_train_loader,
        sim_val_loader,
        device,
        dataset_name="simulated",
        num_epochs=num_epochs,
        lr=1e-3,
    )