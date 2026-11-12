import os
import time

# def debug_log(message):
#     print(
#         f"[{time.strftime('%H:%M:%S')}] "
#         f"PID={os.getpid()} "
#         f"RANK={os.environ.get('RANK', '?')} "
#         f"{message}",
#         flush=True
#     )

# debug_log("Starting train.py")


# debug_log("Importing models")
from .models import VisionTransformer, ActionHead, RoboticActionModel
# debug_log("Importing preprocessing")
from preprocessing.preprocessing import get_dataloaders_oxe
# debug_log("Importing settings")
from preprocessing.utils import get_settings

# debug_log("Other imports")
import torch
import torch.nn.functional as F
from torch.optim import Adam
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from pathlib import Path
import argparse
import numpy as np
import h5py
from datetime import timedelta

N_BINS = 32
BATCH_SIZE = 64
HYDRA = "stanford_hydra_dataset_converted_externally_to_rlds"


def get_model(parameters: dict):
    use_lang = parameters.get("use_lang", False)
    lang_dim = parameters.get("lang_dim", 384) if use_lang else 0
    embed_dim = parameters["embed_dim"]

    vit = VisionTransformer(
        img_size=parameters["img_size"], patch_size=parameters["patch_size"],
        in_channels=parameters["in_channels"], embed_dim=embed_dim,
        num_layers=parameters["num_layers"], num_heads=parameters["num_heads"],
        mlp_dim=parameters["mlp_dim"], dropout=parameters["dropout"],
        use_lang=use_lang, lang_dim=lang_dim,
    )

    head = ActionHead(
        embed_dim=embed_dim + lang_dim if use_lang else embed_dim,
        action_dim=6, use_lang=use_lang, lang_dim=lang_dim
    )

    return RoboticActionModel(vit, {HYDRA: head})


def action_ce_loss(logits, targets):
    return F.cross_entropy(logits.permute(0, 2, 1), targets, label_smoothing=0.05)


def action_accuracy(logits, targets):
    return (logits.argmax(dim=-1) == targets).float().mean().item()


def run_one_epoch(model, loader, device, optimizer=None, use_ddp=False):
    is_train = optimizer is not None
    model.train(is_train)

    base = model.module if use_ddp else model
    use_lang = base.vit.use_lang

    total_loss, total_acc, total_samples = 0.0, 0.0, 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            if is_train:
                optimizer.zero_grad()

            images = batch["image"].to(device)
            language = batch["language_embedding"].to(device) if use_lang else None
            targets = batch["actions"].to(device).long()
            dataset_names = batch["dataset_name"]

            batch_loss, batch_n = 0.0, 0
            for dataset_name in set(dataset_names):
                mask = torch.tensor([n == dataset_name for n in dataset_names], device=device)
                idx = mask.nonzero(as_tuple=True)[0]

                logits = model(
                    images[idx],
                    language[idx] if language is not None else None,
                    dataset_name=dataset_name
                )
                loss = action_ce_loss(logits, targets[idx])
                acc = action_accuracy(logits, targets[idx])

                b = len(idx)
                batch_loss += loss * b
                batch_n += b
                total_loss += loss.item() * b
                total_acc += acc * b
                total_samples += b

            batch_loss /= batch_n
            if is_train:
                batch_loss.backward()
                optimizer.step()

    return total_loss / total_samples, total_acc / total_samples


def train_one_epoch(model, loader, device, optimizer, use_ddp=False):
    return run_one_epoch(model, loader, device, optimizer, use_ddp)


@torch.no_grad()
def evaluate(model, loader, device, use_ddp=False):
    return run_one_epoch(model, loader, device, None, use_ddp)


def setup_distributed_training():

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    print(
        f"[BEFORE INIT] Rank={rank}/{world_size}, "
        f"Local_rank={local_rank}, PID={os.getpid()}",
        flush=True
    )

    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            timeout=timedelta(minutes=30)
        )

    print(
        f"[AFTER INIT] Rank={dist.get_rank()} successfully initialized",
        flush=True
    )

    return local_rank, world_size


def cleanup_distributed_training():
    if dist.is_initialized():
        dist.destroy_process_group()

def train(model, train_loader, val_loader, device, model_filename,
          num_epochs, lr, local_rank, world_size, use_ddp):

    parent = Path(__file__).resolve().parent.parent / "data"
    ckpt_path = parent / f"best_model_{model_filename}.pt"
    metrics_path = parent / f"metrics_epoch_{model_filename}.pt"
    rank = dist.get_rank() if use_ddp else 0

    model = model.to(device)
    if use_ddp:
        model = DDP(model, device_ids=None, output_device=None)

    base_model = model.module if use_ddp else model
    optimizer = Adam(base_model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.1)

    best_val_loss, start_epoch = float("inf"), 0
    metrics = {k: np.array([]) for k in
               ["train_loss", "train_acc", "val_loss", "val_acc"]}

    # Resume
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=device)
        state = {k.removeprefix("module."): v
                 for k, v in ckpt["model_state_dict"].items()}
        base_model.load_state_dict(state)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))

        if metrics_path.is_file():
            with h5py.File(metrics_path, "r") as f:
                metrics = {k: f[k][()][:start_epoch] for k in f}

        if rank == 0:
            print(f"Resuming from epoch {start_epoch}", flush=True)

    # Training
    for epoch in range(start_epoch, num_epochs + 1):
        t_epoch = time.perf_counter()

        tl, ta = train_one_epoch(model, train_loader, device, optimizer, use_ddp)
        vl, va = evaluate(model, val_loader, device, use_ddp)

        if use_ddp:
            vals = torch.tensor([tl, ta, vl, va], dtype=torch.float64,
                                device=device if dist.get_backend() == "nccl" else "cpu")
            dist.all_reduce(vals, op=dist.ReduceOp.SUM)
            tl, ta, vl, va = (vals / world_size).tolist()

        if rank == 0:
            print(f"epoch {epoch:3d} | train {tl:.4f}/{ta:.4f} | val {vl:.4f}/{va:.4f}",
                  flush=True)
            print(
                f"Training + validation time: "
                f"{(time.perf_counter() - t_epoch)/60:.2f} min",
                flush=True
            )

        for k, v in zip(metrics, [tl, ta, vl, va]):
            metrics[k] = np.append(metrics[k], v)

        if rank == 0 and vl < best_val_loss:
            best_val_loss = vl
            tmp_path = ckpt_path.with_suffix(".tmp")
            torch.save({
                "epoch": epoch,
                "model_state_dict": base_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss
            }, tmp_path)
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


    local_rank, world_size = 0, 1
    if args.use_ddp:
        local_rank, world_size = setup_distributed_training()
    print(
    f"Batch size={BATCH_SIZE}, "
    f"Rank={dist.get_rank()}, "
    f"Local_rank={local_rank}, "
    f"World_size={world_size}, "
    f"CPUs={os.getenv('SLURM_CPUS_PER_TASK')}, "
    f"OMP_THREADS={os.getenv('OMP_NUM_THREADS')}, "
    f"Available_CPUs={len(os.sched_getaffinity(0))}, ",
    f"PyTorch threads: {torch.get_num_threads()}\n",
    flush=True
    )
    settings = get_settings(settings=str(Path(__file__).resolve().parent.parent / f"{args.settings}"))
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    torch.set_num_interop_threads(1)

    t0 = time.perf_counter() # keep track of time taken by preprocessing and training.
    train_loader, val_loader, lang_dim = get_dataloaders_oxe(
        cfg=args.settings, n_bins=N_BINS,
        keys_to_keep=settings["keys_to_keep"],
        renaming_keys=settings["renaming_keys"],
        batch_size=BATCH_SIZE,
        data_name_lst=settings["data_name_lst"],
        use_ddp=args.use_ddp,
        local_rank=local_rank,
        world_size=world_size,
        preprocessed_path=args.preprocessed_path,
    )
    if not args.use_ddp or dist.get_rank() == 0: print(f"Preprocessing time: {(time.perf_counter()-t0)/60:.2f} min", flush=True)

    parameters = settings["model_parameters"]
    parameters["lang_dim"] = lang_dim
    parameters["use_lang"] = parameters.get("use_lang", False)

    print(f"{'Vision-Language' if parameters['use_lang'] else 'Vision-only'} Model")

    model = get_model(parameters)
    device = torch.device(f"cuda:{local_rank}" if args.use_ddp and torch.cuda.is_available() else "cpu")

    model, metrics = train(
        model, train_loader, val_loader, device,
        settings["model_filename"], args.n_epochs,
        settings["learning_rate"], local_rank, world_size, args.use_ddp
    )

    if args.use_ddp:
        cleanup_distributed_training()
        if local_rank == 0:
            import shutil
            preprocessed_dir = Path(__file__).resolve().parent.parent / 'preprocessed_data'
            if preprocessed_dir.exists():
                shutil.rmtree(preprocessed_dir)