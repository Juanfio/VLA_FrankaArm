from .models import VisionTransformer, ActionHead, RoboticActionModel
from preprocessing.preprocessing import *

import torch.nn.functional as F
from torch.optim import Adam
from pathlib import Path
import argparse
import numpy as np
import h5py
import os

# DDP imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

N_BINS=32
BATCH_SIZE=32
#################################################
####### Get the data and define the model #######
#################################################
def get_model(parameters: dict):
    #TODO: RENAME THIS FUNCTION. YOU ARE NOT GETTING THE MODEL YOU ARE DEFINING IT BASED ON SOME PARAMETERS.
    """
    Get or define the model based on parameters.

    Args:
        parameters: dict containing model configuration:
            - img_size, patch_size, in_channels, embed_dim, num_layers, num_heads, mlp_dim, dropout
            - use_lang: bool (optional, default False) - whether to use language embeddings
            - lang_dim: int (optional, default 384) - language embedding dimension

    Returns:
        RoboticActionModel: The defined model
    """
    # Extract parameters with defaults for optional ones
    use_lang = parameters.get("use_lang", False)
    lang_dim = parameters.get("lang_dim", 384) if use_lang else 0

    vit = VisionTransformer(
        img_size=parameters["img_size"], patch_size=parameters["patch_size"],
        in_channels=parameters["in_channels"], embed_dim=parameters["embed_dim"],
        num_layers=parameters["num_layers"], num_heads=parameters["num_heads"],
        mlp_dim=parameters["mlp_dim"], dropout=parameters["dropout"],
        use_lang=use_lang, lang_dim=lang_dim,
    )

    # Determine action_head embed_dim based on whether language is used
    # For vision-only: embed_dim = vit.embed_dim
    # For vision-language (concat): embed_dim = vit.embed_dim + lang_dim
    action_head_embed_dim = parameters["embed_dim"] + lang_dim if use_lang else parameters["embed_dim"]

    action_heads = {
        "stanford_hydra_dataset_converted_externally_to_rlds": ActionHead(
            embed_dim=action_head_embed_dim, action_dim=6,
            use_lang=use_lang, lang_dim=lang_dim
        )
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
def run_one_epoch(model, loader, device, optimizer=None, use_ddp: bool = False):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_acc, total_samples = 0.0, 0.0, 0

    # Get base model for accessing attributes (handles DDP wrapper)
    base_model = model.module if use_ddp else model
    vit = base_model.vit
    use_lang = getattr(vit, 'use_lang', False)

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            if is_train:
                optimizer.zero_grad()

            images = batch["image"].to(device)

            if use_lang:
                language = batch["language_embedding"].to(device)
            else:
                # Create dummy language tensor - shape doesn't matter since it's unused
                # But we need a valid tensor to pass through the model
                batch_size = images.shape[0]
                language = torch.zeros(batch_size, 1, device=device)

            targets = batch["actions"].to(device).long()
            dataset_names = batch["dataset_name"]  # list[str], length B

            batch_loss = 0.0
            batch_n = 0

            # the action heads are data-dependent. Then filter data from each dataset before putting it into the model.
            for dataset_name in set(dataset_names):
                mask = torch.tensor([n == dataset_name for n in dataset_names], device=device)
                idx = mask.nonzero(as_tuple=True)[0]

                sub_images = images[idx]
                sub_lang = language[idx]  # works for both use_lang=True/False
                sub_targets = targets[idx]

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


def train_one_epoch(model, loader, device, optimizer, use_ddp: bool = False):
    return run_one_epoch(model, loader, device, optimizer=optimizer, use_ddp=use_ddp)


@torch.no_grad()
def evaluate(model, loader, device, use_ddp: bool = False):
    return run_one_epoch(model, loader, device, optimizer=None, use_ddp=use_ddp)

def build_optimizer(model, lr: float, weight_decay: float = 0.1):
    """
    Pre-training optimizer, per ViT paper Table 3: Adam(beta1=0.9, beta2=0.999).
    Fine-tuning (SGD w/ momentum) is deferred to a later closed-loop stage
    against simulated environment data, not part of this offline pre-training run.
    """
    return Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)


def setup_distributed_training():
    """
    Initialize the distributed process group for DDP.
    Uses 'gloo' backend for CPU-only training.
    Supports both torchrun and srun environment setups.
    Returns: (local_rank, world_size)
    """
    # Support torchrun environment variables (LOCAL_RANK, WORLD_SIZE, RANK)
    # and SLURM variables as fallback
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))

    # torchrun sets MASTER_ADDR and MASTER_PORT automatically
    # If not set (e.g., running manually), use defaults
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    if not dist.is_initialized():
        # Use env:// init_method - torchrun handles the networking
        dist.init_process_group(backend="gloo", init_method="env://")

    return local_rank, world_size


def cleanup_distributed_training():
    """Clean up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def train(
    model,
    train_loader,
    val_loader,
    device,
    model_filename: str,
    num_epochs: int = 20,
    lr: float = 1e-4,
    local_rank: int = 0,
    world_size: int = 1,
    use_ddp: bool = False,
):
    parent_directory = str(Path(__file__).resolve().parent.parent)

    # Wrap with DDP if enabled
    if use_ddp:
        model = DDP(model.to(device), device_ids=None, output_device=None)
        optimizer = build_optimizer(model.module, lr)
    else:
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
    # DDP wraps model in .module, so we need to handle the state dict differently
    checkpoint_path = Path(parent_directory + f'/data/best_model_{model_filename}.pt')
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # DDP adds 'module.' prefix to state dict keys
        if use_ddp:
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint["model_state_dict"].items()}
        else:
            state_dict = checkpoint["model_state_dict"]

        model.load_state_dict(state_dict)

        # For DDP, we use the underlying model's optimizer state
        if use_ddp:
            # Create a dummy optimizer to load state, then recreate with model.module
            dummy_optimizer = build_optimizer(model.module, lr)
            dummy_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            optimizer.load_state_dict(dummy_optimizer.state_dict())
        else:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"]

        print(f"Resuming training from epoch {start_epoch}")

    # update already available metrics_epoch dictionary
    if Path(parent_directory + f'/data/metrics_epoch_{model_filename}.pt').is_file():
        with h5py.File(parent_directory + f'/data/metrics_epoch_{model_filename}.pt', "r") as f:
            metrics_epoch = {key: f[key][()] for key in f}

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, device, optimizer, use_ddp=use_ddp)
        val_loss, val_acc = evaluate(model, val_loader, device, use_ddp=use_ddp)

        # Aggregate validation metrics across all ranks for global metrics
        if use_ddp:
            # Create tensors for all_reduce
            val_loss_tensor = torch.tensor([val_loss], device=device)
            val_acc_tensor = torch.tensor([val_acc], device=device)
            train_loss_tensor = torch.tensor([train_loss], device=device)
            train_acc_tensor = torch.tensor([train_acc], device=device)

            # All reduce to get sum from all ranks
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_acc_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_acc_tensor, op=dist.ReduceOp.SUM)

            # Average by world_size
            val_loss = val_loss_tensor.item() / world_size
            val_acc = val_acc_tensor.item() / world_size
            train_loss = train_loss_tensor.item() / world_size
            train_acc = train_acc_tensor.item() / world_size

        print(
            f"epoch {epoch:3d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        metrics_epoch["train_loss_epoch"] = np.append(metrics_epoch["train_loss_epoch"], train_loss)
        metrics_epoch["train_acc_epoch"] = np.append(metrics_epoch["train_acc_epoch"], train_acc)
        metrics_epoch["val_loss_epoch"] = np.append(metrics_epoch["val_loss_epoch"], val_loss)
        metrics_epoch["val_acc_epoch"] = np.append(metrics_epoch["val_acc_epoch"], val_acc)


        # Only save checkpoint from rank 0 process
        if local_rank == 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                # Get state dict without 'module.' prefix for DDP
                if use_ddp:
                    state_dict = {k.replace('module.', ''): v for k, v in model.state_dict().items()}
                else:
                    state_dict = model.state_dict()

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": state_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    checkpoint_path,
                )
                print(f" -> saved new best checkpoint (val_loss={val_loss:.4f})")

    with h5py.File(parent_directory + f'/data/metrics_epoch_{model_filename}.pt', "w") as f:
        for key, value in metrics_epoch.items():
            f.create_dataset(key, data=value)

    return model, metrics_epoch


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    '''
    Execute from VLA_FrankaArm/ folder.
    Single process: python -m model.train --settings cfg.json --n_epochs 5
    DDP (CPU): torchrun --nproc_per_node=N python -m model.train --settings cfg.json --n_epochs 5 --use_ddp
    With preprocessed data: python -m model.train --settings cfg.json --n_epochs 5 --preprocessed_data preprocessed_data/preprocessed_dataset.pt
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=str)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--use_ddp", action="store_true", help="Enable Distributed Data Parallel")
    parser.add_argument("--preprocessed_path", type=str, default=None,
                        help="Path to preprocessed dataset .pt file (skips expensive preprocessing)")
    args = parser.parse_args()

    # Initialize DDP if enabled
    local_rank = 0
    world_size = 1
    if args.use_ddp:
        local_rank, world_size = setup_distributed_training()
        print(f"DDP initialized: local_rank={local_rank}, world_size={world_size}")

    settings = get_settings(settings=
                                  str(Path(__file__).resolve().parent.parent / f"{args.settings}"))

    data_name_lst = settings["data_name_lst"]
    keys_to_keep = settings["keys_to_keep"]
    renaming_keys = settings["renaming_keys"]
    train_loader, val_loader, lang_dim = get_dataloaders_oxe(
        cfg=args.settings,
        n_bins=N_BINS,
        keys_to_keep=keys_to_keep,
        renaming_keys=renaming_keys,
        batch_size=BATCH_SIZE,
        data_name_lst=data_name_lst,
        use_ddp=args.use_ddp,
        local_rank=local_rank,
        world_size=world_size,
        preprocessed_path=args.preprocessed_path,
        )
        
    parameters = settings["model_parameters"]
    parameters["lang_dim"] = lang_dim

    # Add use_lang parameter (default False for vision-only mode)
    # Set to True to enable vision-language model
    parameters["use_lang"] = parameters.get("use_lang", False)

    if parameters["use_lang"]:
        print("=====================")
        print("Vision-Language Model")
        print("=====================")
    else:
        print("=====================")
        print("Vision-only Model")
        print("=====================")

    model = get_model(parameters)
    # torch.set_num_threads(12)
    # print(f"threads set to: {torch.get_num_threads()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For DDP on CPU, set device based on local_rank if using multiple processes
    # This allows pinning to specific CPU cores for better performance
    if args.use_ddp and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")

    num_epochs=args.n_epochs
    lr=settings["learning_rate"]
    model_filename=settings["model_filename"]
    model, metrics_epoch = train(
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            device=device,
                            model_filename=model_filename,
                            num_epochs=num_epochs,
                            lr=lr,
                            local_rank=local_rank,
                            world_size=world_size,
                            use_ddp=args.use_ddp,
                        )

    # Cleanup DDP process group
    if args.use_ddp:
        cleanup_distributed_training()

    # Clean up preprocessed data directory (created during preprocessing)
    # Only rank 0 should do cleanup to avoid race conditions
    if args.use_ddp and local_rank == 0:
        import shutil
        preprocessed_dir = str(Path(__file__).resolve().parent.parent / 'preprocessed_data')
        if os.path.exists(preprocessed_dir):
            shutil.rmtree(preprocessed_dir)
            print(f"Cleaned up preprocessed data directory: {preprocessed_dir}")