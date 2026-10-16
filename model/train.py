from .models import VisionTransformer, ActionHead, RoboticActionModel
from preprocessing.preprocessing import *

import torch.nn.functional as F
from torch.optim import Adam
from pathlib import Path
import argparse
import numpy as np
import h5py

N_BINS=32
BATCH_SIZE=32
#################################################
####### Get the data and define the model #######
#################################################
def get_model(parameters: dict):
    #TODO: RENAME THIS FUNCTION. YOU ARE NOT GETTING THE MODEL YOU ARE DEFINING IT BASED ON SOME PARAMETERS.
 
    vit = VisionTransformer(
        img_size=parameters["img_size"], patch_size=parameters["patch_size"],
        in_channels=parameters["in_channels"], embed_dim=parameters["embed_dim"],
        num_layers=parameters["num_layers"], num_heads=parameters["num_heads"],
        mlp_dim=parameters["mlp_dim"], dropout=parameters["dropout"], 
        lang_dim=parameters["lang_dim"],
    ) # embed_dim=768, num_layers=12, num_heads=12, mlp_dim=3072

    action_heads = {
        "stanford_hydra_dataset_converted_externally_to_rlds": ActionHead(embed_dim=192, action_dim=6)
        # "taco_play": ActionHead(embed_dim=192, action_dim=6),
        # "simulated": ActionHead(embed_dim=192, action_dim=3), # in simulated environment the action is just the movement of end-effector position (i.e., no rotation).
    }

    model = RoboticActionModel(vit, action_heads)

    return model


#################################################
################ Loss function  #################
#################################################
# ---------------------------------------------------------------------------
# Loss: cross-entropy over discretized action tokens only
# ---------------------------------------------------------------------------
def action_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits:  (B, action_dim, n_bins) -- per-dimension bin logits from ActionHead
    targets: (B, action_dim)          -- discretized bin indices (int64), in [0, n_bins-1]

    nn.functional.cross_entropy expects (B, C, d1, ..., dk) logits and
    (B, d1, ..., dk) integer targets when doing multi-dim classification,
    with C (n_bins here) as dim=1. Our logits have n_bins as the LAST dim,
    so we permute (B, action_dim, n_bins) -> (B, n_bins, action_dim) first.
    """
    logits = logits.permute(0, 2, 1)  # (B, n_bins, action_dim)
    return F.cross_entropy(logits, targets, label_smoothing=0.05)  # mean over batch AND action_dim


def action_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-token (per action-dim, per sample) top-1 accuracy, averaged."""
    preds = logits.argmax(dim=-1)  # (B, action_dim) -> logits.shape = (B, action_dim, n_bins), so argmax gets the argmax over the bins.
    return (preds == targets).float().mean().item()

#################################################
############ Train and Evaluation  ##############
#################################################
'''
* Notes:
    * Using "with torch.set_grad_enabled(is_train):" and "@torch.no_grad()" is redundant but helps documentation.
        * torch.no_grad() disables gradient tracking, same as torch.set_grad_enabled(False)
        * torch.set_grad_enabled(True) enables gradient tracking.
'''
def run_one_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_acc, total_samples = 0.0, 0.0, 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            if is_train:
                optimizer.zero_grad()

            images = batch["image"].to(device)
            language = batch["language_embedding"].to(device)    
            targets = batch["actions"].to(device).long()
            dataset_names = batch["dataset_name"]  # list[str], length B

            batch_loss = 0.0
            batch_n = 0

            # the action heads are data-dependent. Then filter data from each dataset before putting it into the model.
            for dataset_name in set(dataset_names):
                mask = torch.tensor([n == dataset_name for n in dataset_names], device=device)
                idx = mask.nonzero(as_tuple=True)[0]

                sub_images, sub_lang, sub_targets = images[idx], language[idx], targets[idx]

                logits = model(sub_images, sub_lang, dataset_name=dataset_name)
                loss = action_ce_loss(logits, sub_targets)
                acc = action_accuracy(logits, sub_targets)

                b = sub_images.shape[0]
                batch_loss = batch_loss + loss * b
                batch_n += b
                total_loss += loss.item() * b
                total_acc += acc * b
                total_samples += b

            batch_loss = batch_loss / batch_n

            if is_train:
                batch_loss.backward()
                optimizer.step()

    return total_loss / total_samples, total_acc / total_samples


def train_one_epoch(model, loader, device, optimizer):
    return run_one_epoch(model, loader, device, optimizer=optimizer)


@torch.no_grad()
def evaluate(model, loader, device): # 
    return run_one_epoch(model, loader, device, optimizer=None)

def build_optimizer(model, lr: float, weight_decay: float = 0.1):
    """
    Pre-training optimizer, per ViT paper Table 3: Adam(beta1=0.9, beta2=0.999).
    Fine-tuning (SGD w/ momentum) is deferred to a later closed-loop stage
    against simulated environment data, not part of this offline pre-training run.
    """
    return Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)


def train(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs: int = 20,
    lr: float = 1e-4,
    checkpoint_path: str = str(Path(__file__).resolve().parent.parent / 'data/best_model_stanford_hydra.pt'),
):  
    parent_directory = str(Path(__file__).resolve().parent.parent)
    model.to(device)
    optimizer = build_optimizer(model, lr)

    best_val_loss = float("inf")
    
    metrics_epoch = {
                    "train_loss_epoch": np.array([]),
                    "train_acc_epoch": np.array([]),
                    "val_loss_epoch": np.array([]),
                    "val_acc_epoch": np.array([])
                    }
    start_epoch = 0
    
    # start from checkpoint if available.
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"]

        print(f"Resuming training from epoch {start_epoch}")
    
    # update already available metrics_epoch dictionary
    if Path(parent_directory + '/data/metrics_epoch_stanford_hydra.pt').is_file():
        with h5py.File(parent_directory + '/data/metrics_epoch_stanford_hydra.pt', "r") as f:
            metrics_epoch = {key: f[key][()] for key in f}

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"epoch {epoch:3d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        metrics_epoch["train_loss_epoch"] = np.append(metrics_epoch["train_loss_epoch"], train_loss)
        metrics_epoch["train_acc_epoch"] = np.append(metrics_epoch["train_acc_epoch"], train_acc)
        metrics_epoch["val_loss_epoch"] = np.append(metrics_epoch["val_loss_epoch"], val_loss)
        metrics_epoch["val_acc_epoch"] = np.append(metrics_epoch["val_acc_epoch"], val_acc)
  

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            print(f" -> saved new best checkpoint (val_loss={val_loss:.4f})")

    with h5py.File(parent_directory + '/data/metrics_epoch_stanford_hydra.pt', "w") as f:
        for key, value in metrics_epoch.items():
            f.create_dataset(key, data=value)

    return model, metrics_epoch


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    '''
    Execute from VLA_FrankaArm/ folder.
    python -m model.train --settings cfg.json --n_epochs 5
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=str)
    parser.add_argument("--n_epochs", type=int)
    args = parser.parse_args()

    settings = get_settings(settings=
                                  str(Path(__file__).resolve().parent.parent / f"{args.settings}"))    
    
    # data_name_lst = settings["data_name_lst"] # characteristics: Franka, Single Arm, EEF Position.
    # n_episodes = settings["n_episodes"]
    # output_name = settings["output_name"]

    keys_to_keep = settings["keys_to_keep"]
    renaming_keys = settings["renaming_keys"]
    train_loader, val_loader, lang_dim = get_dataloaders_oxe(
        cfg=args.settings, 
        n_bins=N_BINS,
        keys_to_keep=keys_to_keep,
        renaming_keys=renaming_keys,
        batch_size=BATCH_SIZE,
        data_name_lst=["stanford_hydra_dataset_converted_externally_to_rlds"]
        )
        
    parameters = settings["model_parameters"]
    parameters["lang_dim"] = lang_dim
    model = get_model(parameters)
    # torch.set_num_threads(12)
    # print(f"threads set to: {torch.get_num_threads()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs=args.n_epochs
    lr=settings["learning_rate"]
    model, metrics_epoch = train(
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            device=device,
                            num_epochs=num_epochs,
                            lr=lr,
                        )
