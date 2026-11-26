import os
import time
from functools import partial

from .models import VisionTransformer, DinoV2Encoder, ActionHead, RoboticActionModel
from preprocessing.preprocessing import get_dataloaders_oxe, compute_action_class_weights, make_fixed_overfit_loaders
from preprocessing.utils import get_settings

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from pathlib import Path
import argparse
import numpy as np
import h5py
from datetime import timedelta

N_BINS = 32
HYDRA = "stanford_hydra_dataset_converted_externally_to_rlds"


def get_model(parameters: dict):
    use_lang = parameters.get("use_lang", False)
    lang_dim = parameters.get("lang_dim", 384) if use_lang else 0
    encoder = parameters.get("encoder", "vit")  # "vit" (from scratch) or a DINOv2 hub name
    dropout = parameters.get("dropout", 0.0)    # inside the scratch ViT and on the action head's input

    if encoder.startswith("dinov2"):
        vit = DinoV2Encoder(model_name=encoder, freeze=parameters.get("freeze_encoder", True),
                            pooling=parameters.get("pooling", "cls"), use_lang=use_lang,
                            grid_size=parameters.get("grid_size", 4),
                            train_blocks=parameters.get("train_blocks", 0))  # last blocks that still train
        embed_dim = vit.embed_dim  # set by the backbone, not by the config
    else:
        embed_dim = parameters["embed_dim"]
        vit = VisionTransformer(
            img_size=parameters["img_size"], patch_size=parameters["patch_size"],
            in_channels=parameters["in_channels"], embed_dim=embed_dim,
            num_layers=parameters["num_layers"], num_heads=parameters["num_heads"],
            mlp_dim=parameters["mlp_dim"], dropout=dropout,
            use_lang=use_lang, lang_dim=lang_dim,
        )

    # Inverse-dynamics probe: frames t and t+frame_gap -> sum of the actions at t, ..., t+frame_gap-1.
    inverse_dynamics = parameters.get("inverse_dynamics", False)
    frame_input = parameters.get("frame_input", "concat")  # how the probe combines the two frames' features
    if frame_input not in ("concat", "diff"):
        raise ValueError(f"frame_input must be 'concat' or 'diff', got {frame_input!r}.")
    n_frames = 2 if inverse_dynamics else 1
    frame_diff = inverse_dynamics and frame_input == "diff"  # f(t+gap) - f(t) keeps one frame's feature width
    head = ActionHead(embed_dim=embed_dim * (1 if frame_diff else n_frames), action_dim=6, use_lang=use_lang,
                      lang_dim=lang_dim, dropout=dropout,
                      n_tokens=getattr(vit, "n_tokens", 1))  # several tokens: pooled by attention
    return RoboticActionModel(vit, {HYDRA: head}, n_frames=n_frames, frame_diff=frame_diff)


def action_ce_loss(logits, targets, class_weights=None, label_smoothing=0.0):
    """Cross-entropy averaged over the action dimensions, with optional per-dimension class weights."""
    return torch.stack([
        F.cross_entropy(logits[:, d, :], targets[:, d], label_smoothing=label_smoothing,
                        weight=None if class_weights is None else class_weights[d])
        for d in range(targets.shape[1])
    ]).mean()


def action_accuracy(logits, targets):
    return (logits.argmax(dim=-1) == targets).float().mean().item()


def run_one_epoch(model, loader, device, loss_fn, optimizer=None, scheduler=None):
    """One pass over the loader, training when an optimizer is given. Returns (mean loss, mean accuracy)."""
    is_train = optimizer is not None
    model.train(is_train)
    use_lang = getattr(model, "module", model).vit.use_lang  # unwrap DDP
    total_loss, total_acc, total_samples = 0.0, 0.0, 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            images = batch["image"].to(device)  # images, or the cached features of a frozen encoder
            language = batch["language_embedding"].to(device) if use_lang else None
            targets = batch["actions"].to(device).long()
            dataset_names = batch["dataset_name"]

            batch_loss, batch_n = 0.0, 0
            for dataset_name in set(dataset_names):  # one action head per dataset
                idx = torch.tensor([n == dataset_name for n in dataset_names], device=device).nonzero(as_tuple=True)[0]
                logits = model(images[idx], language[idx] if language is not None else None, dataset_name=dataset_name)
                loss = loss_fn(logits, targets[idx])
                b = len(idx)
                batch_loss += loss * b
                batch_n += b
                total_loss += loss.item() * b
                total_acc += action_accuracy(logits, targets[idx]) * b
                total_samples += b

            if is_train:
                optimizer.zero_grad()
                (batch_loss / batch_n).backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()  # the learning rate changes every batch

    return total_loss / total_samples, total_acc / total_samples


@torch.no_grad()
def diagnose_action_bins(model, loader, device, n_actions=6, n_bins=N_BINS):
    # TODO: MOVE THIS FUNCTION TO PREPROCESSING.UTILS -> It's for diagnosing not training.
    """
    Checks whether the model predicts only a subset of the action bins present in the targets, and compares
    exact and ±1-bin accuracy with the best constant prediction (majority baselines). A ±1-bin hit counts
    neighbouring bins as correct. Aggregates statistics across all DDP processes.
    """
    is_ddp = dist.is_available() and dist.is_initialized()
    base_model = getattr(model, "module", model)  # avoids DDP forward synchronization during evaluation
    base_model.eval()

    # Integer counts, accumulated locally on each process.
    target_counts = torch.zeros(n_actions, n_bins, dtype=torch.long)
    pred_counts = torch.zeros(n_actions, n_bins, dtype=torch.long)
    correct = torch.zeros(2, n_actions, dtype=torch.long)  # rows: exact bin, within ±1 bin

    for batch in loader:
        language = batch["language_embedding"] if base_model.vit.use_lang else None
        for dataset_name in set(batch["dataset_name"]):
            idx = torch.tensor([n == dataset_name for n in batch["dataset_name"]]).nonzero(as_tuple=True)[0]
            logits = base_model(batch["image"][idx].to(device),
                                language[idx].to(device) if language is not None else None,
                                dataset_name=dataset_name)
            p, t = logits.argmax(dim=-1).cpu(), batch["actions"][idx].long()
            error = (p - t).abs()
            correct += torch.stack([(error == 0).sum(0), (error <= 1).sum(0)])
            for d in range(n_actions):
                target_counts[d] += torch.bincount(t[:, d], minlength=n_bins)
                pred_counts[d] += torch.bincount(p[:, d], minlength=n_bins)

    if is_ddp:  # sum the raw counts of all processes (NCCL requires CUDA tensors; Gloo can use CPU tensors)
        size = n_actions * n_bins
        stats = torch.cat([target_counts.flatten(), pred_counts.flatten(), correct.flatten()])
        stats = stats.to(device if dist.get_backend() == "nccl" else "cpu")
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        target_counts, pred_counts, correct = stats.cpu().split([size, size, 2 * n_actions])
        target_counts, pred_counts = target_counts.view(n_actions, n_bins), pred_counts.view(n_actions, n_bins)
        correct = correct.view(2, n_actions)

    # Calculate metrics using globally aggregated counts.
    results = {}
    for d in range(n_actions):
        total = target_counts[d].sum().item()
        if total == 0:
            continue
        predicted, present = pred_counts[d] > 0, target_counts[d] > 0
        padded = F.pad(target_counts[d], (1, 1))
        results[d] = {
            "accuracy": correct[0, d].item() / total,
            "majority_baseline": target_counts[d].max().item() / total,
            "accuracy_within_1_bin": correct[1, d].item() / total,
            "majority_baseline_within_1_bin": (padded[:-2] + padded[1:-1] + padded[2:]).max().item() / total,
            "unique_predicted_bins": predicted.sum().item(),  # how many different bins were predicted
            "unique_target_bins": present.sum().item(),  # how many different bins could have been predicted
            "target_bin_coverage": (predicted & present).sum().item() / present.sum().item(),  # fraction predicted
            "most_common_predicted_bin": pred_counts[d].argmax().item(),
        }
        if not is_ddp or dist.get_rank() == 0:
            print(f"Action {d}: " + ", ".join(f"{k.replace('_', ' ')}={v:.3f}" if isinstance(v, float)
                                              else f"{k.replace('_', ' ')}={v}" for k, v in results[d].items()),
                  flush=True)
    if results and (not is_ddp or dist.get_rank() == 0):  # the mean is comparable with the per-epoch accuracy
        means = {k: np.mean([r[k] for r in results.values()]) for k in list(next(iter(results.values())))[:4]}
        print("Mean over actions: " + ", ".join(f"{k.replace('_', ' ')}={v:.3f}" for k, v in means.items()), flush=True)
    return results


def setup_distributed_training():

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            timeout=timedelta(minutes=30)
        )

    return local_rank, world_size


def cleanup_distributed_training():
    if dist.is_initialized():
        dist.destroy_process_group()


def train(model, train_loader, val_loader, device, settings, num_epochs, world_size, use_ddp, class_weights=None):
    """
    Train with AdamW, log per-epoch metrics, optionally save the best checkpoint (and resume from it).
    Config keys: learning_rate, weight_decay, label_smoothing, save_checkpoint, model_filename, and
    lr_schedule: "cosine" warms up over the first warmup_frac of all steps, then decays; otherwise constant.
    """
    parent = Path(__file__).resolve().parent.parent / "data"
    ckpt_path = parent / f"best_model_{settings['model_filename']}.pt"
    metrics_path = parent / f"metrics_epoch_{settings['model_filename']}.pt"
    rank = dist.get_rank() if use_ddp else 0

    model = model.to(device)
    if class_weights is not None: class_weights = class_weights.to(device)
    if use_ddp: model = DDP(model, device_ids=None, output_device=None)

    base_model = model.module if use_ddp else model
    loss_fn = partial(action_ce_loss, class_weights=class_weights, label_smoothing=settings.get("label_smoothing", 0.0))
    lr = settings["learning_rate"]
    # Fine-tuned DINOv2 blocks get their own, usually smaller, learning rate: encoder_learning_rate.
    encoder = [p for p in base_model.vit.parameters() if p.requires_grad] if isinstance(base_model.vit, DinoV2Encoder) else []
    groups = [{"params": [p for p in base_model.parameters() if all(p is not q for q in encoder)], "lr": lr}]
    if encoder:
        groups.append({"params": encoder, "lr": settings.get("encoder_learning_rate", lr)})
    optimizer = AdamW(groups, betas=(0.9, 0.999), weight_decay=settings.get("weight_decay", 0.01))
    scheduler = None
    if settings.get("lr_schedule") == "cosine":  # epochs run from 0 to num_epochs inclusive
        scheduler = OneCycleLR(optimizer, max_lr=[g["lr"] for g in groups], total_steps=(num_epochs + 1) * len(train_loader),
                               pct_start=settings.get("warmup_frac", 0.05), cycle_momentum=False)

    best_val_loss, start_epoch = float("inf"), 0
    metrics = {k: np.array([]) for k in
               ["train_loss", "train_acc", "val_loss", "val_acc"]}

    # Resume
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=device)
        base_model.load_state_dict({k.removeprefix("module."): v for k, v in ckpt["model_state_dict"].items()})
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))

        # Only rank 0 reads the HDF5 metrics file.
        if rank == 0 and metrics_path.is_file():
            with h5py.File(metrics_path, "r") as f:
                metrics = {k: f[k][()][:start_epoch] for k in f}

        # Share the loaded metrics with all other processes.
        if use_ddp:
            metrics_obj = [metrics if rank == 0 else None]
            dist.broadcast_object_list(metrics_obj, src=0)
            metrics = metrics_obj[0]

        if rank == 0:
            print(f"Resuming from epoch {start_epoch}", flush=True)

    # Training
    for epoch in range(start_epoch, num_epochs + 1):
        t_epoch = time.perf_counter()
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)  # draw a different random set of training samples every epoch
        epoch_lr = optimizer.param_groups[0]["lr"]

        tl, ta = run_one_epoch(model, train_loader, device, loss_fn, optimizer, scheduler)
        vl, va = run_one_epoch(model, val_loader, device, loss_fn)

        if use_ddp:
            vals = torch.tensor([tl, ta, vl, va], dtype=torch.float64,
                                device=device if dist.get_backend() == "nccl" else "cpu")
            dist.all_reduce(vals, op=dist.ReduceOp.SUM)
            tl, ta, vl, va = (vals / world_size).tolist()

        if rank == 0:
            print(f"epoch {epoch:3d} | train {tl:.4f}/{ta:.4f} | val {vl:.4f}/{va:.4f} | lr {epoch_lr:.2e} | "
                  f"{(time.perf_counter() - t_epoch) / 60:.2f} min", flush=True)

        for k, v in zip(metrics, [tl, ta, vl, va]):
            metrics[k] = np.append(metrics[k], v)

        if settings.get("save_checkpoint", False) and rank == 0 and vl < best_val_loss:
            best_val_loss = vl
            tmp_path = ckpt_path.with_suffix(".tmp")
            torch.save({"epoch": epoch, "model_state_dict": base_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                        "val_loss": best_val_loss}, tmp_path)
            os.replace(tmp_path, ckpt_path)
            print(f" -> saved best checkpoint ({vl:.4f})", flush=True)

    # Save metrics only on global rank 0
    if rank == 0:
        with h5py.File(metrics_path, "w") as f:
            for k, v in metrics.items():
                f.create_dataset(k, data=v)

    if use_ddp:
        dist.barrier()

    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=str)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--use_ddp", action="store_true")
    parser.add_argument("--preprocessed_path", type=str, default=None)
    args = parser.parse_args()

    # ─────────────────────────────────────────────────────────────────────────────
    # Initialize DDP & open settings.
    # ─────────────────────────────────────────────────────────────────────────────
    local_rank, world_size = setup_distributed_training() if args.use_ddp else (0, 1)
    rank = dist.get_rank() if args.use_ddp else 0
    settings = get_settings(settings=str(Path(__file__).resolve().parent.parent / f"{args.settings}"))
    batch_size = settings.get("batch_size", 64)  # per process: the global batch is batch_size * world_size
    print(f"Batch size={batch_size} (global {batch_size * world_size}), Rank={rank}, Local_rank={local_rank}, "
          f"World_size={world_size}, CPUs={os.getenv('SLURM_CPUS_PER_TASK')}, "
          f"OMP_THREADS={os.getenv('OMP_NUM_THREADS')}, Available_CPUs={len(os.sched_getaffinity(0))}, "
          f"PyTorch threads: {torch.get_num_threads()}\n", flush=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", torch.get_num_threads())))
    torch.set_num_interop_threads(1)

    # ─────────────────────────────────────────────────────────────────────────────
    # Prepare model & data.
    # ─────────────────────────────────────────────────────────────────────────────
    parameters = settings["model_parameters"]
    parameters["use_lang"] = parameters.get("use_lang", False)
    # Inverse-dynamics probe: frames t and t+frame_gap -> sum of the actions at t, ..., t+frame_gap-1 (0: no probe).
    frame_gap = int(parameters.get("frame_gap", 1)) if parameters.get("inverse_dynamics", False) else 0
    if parameters.get("inverse_dynamics", False) and frame_gap < 1:
        raise ValueError(f"frame_gap must be at least 1, got {frame_gap}.")
    if rank == 0:
        combined = ("feature difference" if parameters.get("frame_input", "concat") == "diff"
                    else "concatenated features")
        target = "action at t" if frame_gap == 1 else f"sum of the actions at t..t+{frame_gap - 1}"
        print(f"{'Vision-Language' if parameters['use_lang'] else 'Vision-only'} Model"
              + (f", inverse-dynamics probe: frames t and t+{frame_gap} ({combined}) -> {target}"
                 if frame_gap else ""), flush=True)
    model = get_model(parameters)  # before the data, so that preprocessing can use a frozen encoder
    device = torch.device(f"cuda:{local_rank}" if args.use_ddp and torch.cuda.is_available() else "cpu")

    # A frozen encoder gives the same features in every epoch: preprocessing replaces each episode's frames by
    # their features right after reading them (never holding all images), then training only updates the head
    # (with train_blocks > 0: the features are the tokens after the frozen blocks, and the last blocks train too).
    frozen = isinstance(model.vit, DinoV2Encoder) and model.vit.freeze
    t0 = time.perf_counter() # keep track of time taken by preprocessing and training.
    train_loader, val_loader, lang_dim = get_dataloaders_oxe(
        cfg=args.settings, n_bins=N_BINS,
        keys_to_keep=settings["keys_to_keep"],
        renaming_keys=settings["renaming_keys"],
        batch_size=batch_size,
        data_name_lst=settings["data_name_lst"],
        use_ddp=args.use_ddp,
        local_rank=local_rank,
        world_size=world_size,
        preprocessed_path=args.preprocessed_path,
        image_encoder=model.vit.to(device) if frozen else None,
        frame_gap=frame_gap,  # samples hold frames t and t+frame_gap, targets the summed actions in between
    )
    if rank == 0:
        # # do features change between consecutive frames?
        # ds = val_loader.dataset.dataset                        # all samples, in time order
        # n = 20_000                                             # ~30 episodes is enough
        # f = ds.images[:n].float().flatten(1)
        # ep = (ds.dataset_ids * (int(ds.episode_indices.max()) + 1) + ds.episode_indices)[:n]
        # print(f"random pairs: mean cosine {F.cosine_similarity(f, f[torch.randperm(n)], dim=1).mean():.4f}")
        # for k in (1, 3, 10, 30):
        #     same = ep[k:] == ep[:-k]
        #     cos = F.cosine_similarity(f[k:][same], f[:-k][same], dim=1)
        #     print(f"gap {k:2d}: mean cosine {cos.mean():.4f}, identical {(cos > 0.9999).float().mean():.1%}")
        print(f"Preprocessing time: {(time.perf_counter() - t0) / 60:.2f} min, "
                        f"inputs {tuple(train_loader.dataset.dataset.images.shape)}", flush=True)
    if parameters["use_lang"] and lang_dim != parameters.get("lang_dim", 384):
        raise ValueError(f"Language embeddings have {lang_dim} values, the model expects {parameters.get('lang_dim', 384)}.")

    # Memorization sanity check: train and evaluate on the same few fixed training samples.
    if settings.get("overfit_samples"):
        train_loader, val_loader = make_fixed_overfit_loaders(
            train_loader, n_samples=settings["overfit_samples"], batch_size=16, seed=42)

    # Class weights from the training targets (all DDP processes must execute this function).
    class_weights = None
    if settings.get("class_weights", False):
        class_weights, train_bin_counts = compute_action_class_weights(train_loader, n_bins=N_BINS, n_actions=6)
        if rank == 0:
            print("\n===== ACTION CLASS WEIGHTS =====")
            for d in range(6):
                observed = train_bin_counts[d] > 0
                print(f"Action {d}: observed bins={observed.sum().item()}/{N_BINS}, "
                      f"min weight={class_weights[d][observed].min():.3f}, max weight={class_weights[d].max():.3f}")

    # ─────────────────────────────────────────────────────────────────────────────
    # Train model
    # ─────────────────────────────────────────────────────────────────────────────
    model, metrics = train(model, train_loader, val_loader, device, settings, args.n_epochs,
                           world_size, args.use_ddp, class_weights=class_weights)

    # Diagnose action-bin distributions across all validation shards.
    if rank == 0: print("Running action-bin diagnostics...", flush=True)
    diagnose_action_bins(model, val_loader, device)

    if args.use_ddp:
        cleanup_distributed_training()
        if local_rank == 0:
            import shutil
            preprocessed_dir = Path(__file__).resolve().parent.parent / 'preprocessed_data'
            if preprocessed_dir.exists():
                shutil.rmtree(preprocessed_dir)
