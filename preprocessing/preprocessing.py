import torch

from torch.utils.data import Dataset, DataLoader, Subset

from preprocessing.utils import get_settings

import json
from pathlib import Path
import numpy as np


N_BINS = 32
# Normalization constants DINOv2 was pretrained with (ImageNet RGB statistics).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "language_models" / "all-MiniLM-L6-v2"

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def open_combined_dataset(path_data, oxe_dataset_name, data_name_lst=None, image_encoder=None):
    """
    Load combined dataset from HDF5, returns list[dict] and metadata. With image_encoder, each episode's frames
    are replaced by their features (see encode_frames) right after reading, so the images of all episodes
    are never in memory together.
    """
    import h5py
    data, metadata = [], {}
    with h5py.File(f"{path_data}/{oxe_dataset_name}.h5", "r") as f:
        names = data_name_lst if data_name_lst else f.keys()
        for data_name in names:
            if data_name not in f:
                print(f"Warning: Dataset '{data_name}' not found. Skipping.")
                continue
            grp = f[data_name]
            metadata[data_name] = {"actions_metadata": json.loads(grp.attrs["actions_metadata"])}
            print(f"n_episodes: {len(grp.keys())}")
            for ep_idx, ep_key in enumerate(grp.keys()):
                g = grp[ep_key]
                images = {k: g["images"][k][:] if image_encoder is None else encode_frames(g["images"][k][:], image_encoder)
                          for k in g["images"].keys()}
                actions = {k: g["actions"][k][:] for k in g["actions"].keys()}
                lang = {k: g["language"][k][:] for k in g["language"].keys()}
                n_steps = next(iter(actions.values())).shape[0]
                for t in range(n_steps):
                    sample = {k: v[t][:-1] if v.ndim == 2 else v[t] for k, v in actions.items()}
                    sample.update({k: v[t] for k, v in images.items()})
                    sample.update({k: v[t] for k, v in lang.items()})
                    sample["episode_idx"], sample["dataset_name"] = ep_idx, data_name
                    data.append(sample)
            meta = metadata[data_name]
            meta["image_keys"] = list(g["images"].keys())
            meta["action_keys"] = list(g["actions"].keys())
            meta["language_keys"] = list(g["language"].keys())
            meta["total_samples"] = n_steps * len(grp.keys())
    return data, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Discretization
# ─────────────────────────────────────────────────────────────────────────────

def discretize(values, p1, p99, n_bins=N_BINS):
    """Map continuous values to discrete bins [0, n_bins-1]."""
    values = values.reshape(-1, values.shape[-1] if values.ndim > 1 else 1)
    discretized = np.zeros_like(values, dtype=np.int64)
    for d in range(values.shape[1]):
        edges = np.linspace(p1[d], p99[d], n_bins + 1) # equally spaced bins. Attention: it creates class imbalance.
        discretized[:, d] = np.clip(np.digitize(values[:, d], edges[1:-1], right=True), 0, n_bins - 1)
    return discretized.squeeze()


def undiscretize(bin_indices, p1, p99, n_bins=N_BINS):
    """Inverse of discretize: map bin indices back to continuous values."""
    if isinstance(bin_indices, np.ndarray): bin_indices = torch.from_numpy(bin_indices)
    if isinstance(p1, np.ndarray): p1 = torch.from_numpy(p1)
    if isinstance(p99, np.ndarray): p99 = torch.from_numpy(p99)
    return p1 + (bin_indices.float() + 0.5) * (p99 - p1) / n_bins


def apply_discretization(data, metadata, n_bins=N_BINS):
    """Apply discretization per dataset, return (discretized_data, percentiles)."""
    action_percentiles, data_discretized = {}, []
    for data_name in metadata:
        keys = metadata[data_name]["action_keys"]
        batch = [s for s in data if s["dataset_name"] == data_name]
        action_percentiles[data_name] = {}
        batch_disc = [dict(s) for s in batch]
        for key in keys:
            stacked = np.stack([s[key] for s in batch])
            if stacked.ndim == 1: stacked = stacked.reshape(-1, 1)
            p1, p99 = np.percentile(stacked, [1, 99], axis=0)
            action_percentiles[data_name][key] = {"p1": p1, "p99": p99}
            disc = discretize(stacked, p1, p99, n_bins)
            for s, d in zip(batch_disc, disc.reshape(len(batch), -1)):
                s[key] = d.squeeze()
        data_discretized.extend(batch_disc)
    return data_discretized, action_percentiles



def compute_action_class_weights(train_loader, n_bins=32, n_actions=6, max_weight=2.0, alpha=0.25):
    """
    Inverse-frequency class weights from the training targets, without iterating through the DataLoader:
    per action dimension, (mean bin frequency / bin frequency) ** alpha clamped at max_weight
    (alpha=0.5 is the square root, larger alpha weights rare bins more); bins that never occur get 0.
    Supports SharedMemoryRobotDataset and regular Subset datasets. In DDP, every rank has the same
    training subset, so no all_reduce is necessary. Returns (class_weights, bin_counts), both (n_actions, n_bins).
    """
    dataset = train_loader.dataset
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    if isinstance(getattr(base, "actions", None), torch.Tensor):  # direct access in SharedMemoryRobotDataset
        targets = base.actions if base is dataset else base.actions.index_select(
            0, torch.as_tensor(dataset.indices, dtype=torch.long))
    else:  # fallback for regular Python datasets
        targets = torch.as_tensor(np.stack([dataset[i]["actions"] for i in range(len(dataset))]))
    targets = targets.reshape(-1, n_actions).long().cpu()
    if torch.any((targets < 0) | (targets >= n_bins)):
        raise ValueError(f"Action targets must be in [0, {n_bins - 1}]")

    bin_counts = torch.stack([torch.bincount(targets[:, d], minlength=n_bins) for d in range(n_actions)])
    class_weights = torch.zeros(n_actions, n_bins, dtype=torch.float32)
    for d in range(n_actions):
        observed = bin_counts[d] > 0  # bins that occur at least once
        if observed.any():
            frequencies = bin_counts[d, observed].float()
            class_weights[d, observed] = (frequencies.mean() / frequencies).pow(alpha).clamp(max=max_weight)
    return class_weights, bin_counts



# ─────────────────────────────────────────────────────────────────────────────
# Filtering & Renaming
# ─────────────────────────────────────────────────────────────────────────────

def filter_keys(data, keys_to_keep, renaming_keys=None):
    """Filter samples to specified keys and optionally rename them."""
    renaming_keys = renaming_keys or {}
    filtered = []
    for s in data:
        keys = keys_to_keep[s["dataset_name"]]
        sample = {k: s[k] for k in keys}
        for old, new in renaming_keys.get(s["dataset_name"], []):
            sample[new] = sample.pop(old)
        filtered.append(sample)
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Image Processing
# ─────────────────────────────────────────────────────────────────────────────

def episode_to_ids(dataset):
    """Map (dataset_name, episode_idx) to list of sample indices."""
    episode_to_indices = {}
    for i, s in enumerate(dataset):
        key = (s["dataset_name"], s["episode_idx"])
        episode_to_indices.setdefault(key, []).append(i)
    return episode_to_indices


def prepare_frames(frames, out_dim=224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Frames (T, H, W, C), uint8 (scaled to [0, 1]) or float -> resized, normalized tensor (T, C, out_dim, out_dim)."""
    from torchvision import transforms
    import torchvision.transforms.functional as TF
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).contiguous()  # standard NCHW layout
    x = x.float().div(255) if x.dtype == torch.uint8 else x.float()
    x = transforms.Resize((out_dim, out_dim), interpolation=transforms.InterpolationMode.BILINEAR)(x)
    return TF.normalize(x, mean, std)


@torch.no_grad()
def encode_frames(frames, image_encoder, batch_size=256):
    """Frames (T, H, W, C) -> what a frozen DinoV2Encoder caches per frame (its cache_features), batch by batch."""
    device = next(image_encoder.parameters()).device
    return torch.cat([image_encoder.cache_features(prepare_frames(frames[i:i + batch_size]).to(device)).cpu()
                      for i in range(0, len(frames), batch_size)])


def image_resizing(data, out_dim=224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Resize images to out_dim x out_dim and normalize them per channel. By default with the fixed ImageNet
    mean/std, which the pretrained DINOv2 encoder expects and which are also available at rollout time.
    mean=None and std=None restore the old behaviour (statistics per (dataset, episode)): that only suits the
    from-scratch ViT and cannot be reproduced at rollout, where the episode's statistics are unknown in advance.
    """
    if (mean is None) != (std is None):
        raise ValueError("Pass both mean and std, or neither (per-episode statistics).")
    if mean is not None:
        for s in data:
            s["image"] = prepare_frames(s["image"][None], out_dim, mean, std)[0]
        return data

    from torchvision import transforms
    import torchvision.transforms.functional as TF
    resize = transforms.Resize((out_dim, out_dim), interpolation=transforms.InterpolationMode.BILINEAR)
    for indices in episode_to_ids(data).values():
        stack = torch.stack([TF.to_tensor(data[i]["image"]) for i in indices])
        ep_mean, ep_std = stack.mean(dim=[0, 2, 3]), stack.std(dim=[0, 2, 3]).clamp(min=1e-6)
        for i, img in zip(indices, stack):
            data[i]["image"] = TF.normalize(resize(img), ep_mean, ep_std)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Language Embeddings
# ─────────────────────────────────────────────────────────────────────────────

def _to_str(x):
    if isinstance(x, np.ndarray): x = x.item()
    return x.decode("utf-8") if isinstance(x, bytes) else str(x)


def load_text_model(name="all-MiniLM-L6-v2", local_dir=LOCAL_MODEL_DIR):
    
    from sentence_transformers import SentenceTransformer # only rank 0 performs preprocessing and needs this.
    
    if (local_dir / "modules.json").exists(): return SentenceTransformer(str(local_dir))
    model = SentenceTransformer(name)
    local_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(local_dir))
    return model


def add_language_embeddings(data, model_name="all-MiniLM-L6-v2", out_key="language_embedding"):
    texts = [_to_str(s["language_instruction"]) for s in data]
    uniq = sorted(set(texts))
    print(f"{len(uniq)} unique instructions")
    model = load_text_model(model_name)
    emb = model.encode(uniq, convert_to_tensor=True, normalize_embeddings=True).cpu()
    lookup = {t: e for t, e in zip(uniq, emb)}
    for s, t in zip(data, texts):
        s[out_key] = lookup[t].clone()
    return data, emb.shape[1], model


# ─────────────────────────────────────────────────────────────────────────────
# Episode Splitting
# ─────────────────────────────────────────────────────────────────────────────
def split_by_episode(dataset, val_frac=0.1, seed=42):
    """
    Split whole episodes into train/validation sets (val_frac of the episodes, at least one in each set).
    For SharedMemoryRobotDataset, NumPy operations avoid constructing large Python dictionaries and lists.

    Returns:
        train_idx, val_idx: NumPy arrays for SharedMemoryRobotDataset, Python lists otherwise.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1.")
    rng = np.random.default_rng(seed)

    if isinstance(dataset, SharedMemoryRobotDataset):
        ds, ep = dataset.dataset_ids.numpy(), dataset.episode_indices.numpy()  # NumPy views (no data copy)
        if len(ds) == 0:
            raise ValueError("Cannot split an empty dataset.")
        # Map every sample to its episode via unique (dataset_id, episode_id) keys (episode IDs restart per dataset).
        unique_keys, sample_episode = np.unique(ds * (int(ep.max()) + 1) + ep, return_inverse=True)
        episodes = range(len(unique_keys))
    else:
        ep_to_idx = episode_to_ids(dataset)
        episodes = sorted(ep_to_idx.keys())

    if len(episodes) < 2:
        raise ValueError("At least two episodes are required for a train/validation split.")
    n_val = min(max(1, int(val_frac * len(episodes))), len(episodes) - 1)
    order = rng.permutation(len(episodes))  # random episode order: the first n_val episodes validate

    if isinstance(dataset, SharedMemoryRobotDataset):
        val_mask = np.isin(sample_episode, order[:n_val])
        return np.flatnonzero(~val_mask), np.flatnonzero(val_mask)
    shuffled = [episodes[i] for i in order]
    return ([i for e in shuffled[n_val:] for i in ep_to_idx[e]],
            [i for e in shuffled[:n_val] for i in ep_to_idx[e]])


# ─────────────────────────────────────────────────────────────────────────────
# Data Loaders
# ─────────────────────────────────────────────────────────────────────────────

def worker_init_fn(worker_id):
    torch.set_num_threads(1)


class SharedMemoryRobotDataset(Dataset):
    """Dataset backed by tensors, memory-mapped from disk when several DDP processes share them."""

    def __init__(self, shared_data, indices=None):
        self.__dict__.update(shared_data)
        self.indices = (
            indices if indices is not None
            else range(len(self.images))
        )

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        actual = self.indices[idx]
        return {
            "image": self.images[actual],
            "actions": self.actions[actual],
            "language_embedding": self.language_embeddings[actual],
            "dataset_id": self.dataset_ids[actual],
            "episode_idx": self.episode_indices[actual],
            "dataset_name": self.dataset_names[self.dataset_ids[actual].item()],
        }


class DistributedWeightedSampler:
    """Weighted sampling per DDP rank with reproducibility."""

    def __init__(self, weights, num_samples, num_replicas, rank,
                 replacement=True, seed=42, shuffle=True):
        self.w = torch.as_tensor(weights, dtype=torch.double)
        self.n, self.nr, self.rank = num_samples, num_replicas, rank
        self.replacement, self.seed, self.shuffle = replacement, seed, shuffle
        self.epoch = 0

    def set_epoch(self, e): self.epoch = e

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch + self.rank * 1000)
        if self.replacement:
            idx = torch.multinomial(self.w, self.n, replacement=True, generator=g).tolist()
        else:
            _, idx = torch.topk(self.w * torch.rand(len(self.w), generator=g), min(self.n, len(self.w)))
            idx = idx.tolist()
        if self.shuffle:
            idx = [idx[i] for i in torch.randperm(len(idx), generator=g).tolist()]
        return iter(idx)

    def __len__(self): return self.n


def get_dataloaders_oxe(cfg, n_bins=N_BINS, keys_to_keep=None, renaming_keys=None,
                        batch_size=32, data_name_lst=None, use_ddp=False,
                        local_rank=0, world_size=1, preprocessed_path=None, image_encoder=None):
    """
    Build OXE train/val dataloaders, single-process or DDP. Only global rank 0 preprocesses. A single process
    keeps the result in memory; with several processes, or when preprocessed_path is given, it is written to
    disk and memory-mapped by every process. An existing preprocessed_path is reused instead of preprocessing.
    With image_encoder (a frozen DinoV2Encoder) the dataset holds its cache_features output instead of images,
    computed episode by episode while reading.
    """
    import torch.distributed as dist
    settings = get_settings(settings=str(Path(__file__).resolve().parent.parent / f"{cfg}"))
    oxe_name, path_data = settings["oxe_dataset_name"], settings["path_data"]
    if data_name_lst is None: data_name_lst = settings.get("data_name_lst")
    keys_to_keep = keys_to_keep or settings["keys_to_keep"]
    renaming_keys = renaming_keys or settings["renaming_keys"]
    rank = dist.get_rank() if use_ddp else 0
    dataset_path = Path(preprocessed_path or "./preprocessed_data/preprocessed_dataset.pt")

    features = None if image_encoder is None else image_encoder.name  # None: the dataset holds images

    loaded = None
    if rank == 0 and not (preprocessed_path and dataset_path.exists()):
        data, metadata = open_combined_dataset(path_data, oxe_name, data_name_lst, image_encoder)
        print(f"Total samples: {len(data)}", flush=True)
        data, action_pct = apply_discretization(data, metadata, n_bins)
        data = filter_keys(data, keys_to_keep, renaming_keys)
        if image_encoder is None:
            data = image_resizing(data)
        data, lang_dim, _ = add_language_embeddings(data)
        loaded = {**_build_shared_data(data, metadata), "metadata": metadata, "action_percentiles": action_pct,
                  "lang_dim": lang_dim, "image_normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
                  "image_features": features}
        del data  # release the per-sample copies
        if world_size > 1 or preprocessed_path:
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(loaded, dataset_path)
            print(f"Preprocessed data saved to {dataset_path}", flush=True)
            loaded = None  # release memory: rank 0 memory-maps the file below, like every other rank
    if use_ddp:
        dist.barrier()  # wait until rank 0 has finished writing
    if loaded is None:
        loaded = torch.load(str(dataset_path), map_location="cpu", mmap=True, weights_only=False)

    # Files written before the switch to fixed ImageNet normalization lack this key
    # and hold per-episode-normalized images, which the DINOv2 encoder must not see.
    norm = loaded.get("image_normalization")
    if norm is None or tuple(norm["mean"]) != IMAGENET_MEAN or tuple(norm["std"]) != IMAGENET_STD:
        raise ValueError(f"{dataset_path} was preprocessed with image normalization {norm}, expected mean "
                         f"{IMAGENET_MEAN} and std {IMAGENET_STD}. Delete it or run without --preprocessed_path.")
    if loaded.get("image_features") != features:
        raise ValueError(f"{dataset_path} holds {loaded.get('image_features') or 'images'}, but this run needs "
                         f"{features or 'images'}. Delete it or run without --preprocessed_path.")

    dataset = SharedMemoryRobotDataset(loaded)
    train_idx, val_idx = split_by_episode(dataset)
    return (_make_dataloader(dataset, train_idx, world_size, rank, batch_size, True),
            _make_dataloader(dataset, val_idx, world_size, rank, batch_size, False), loaded["lang_dim"])


def _build_shared_data(data, metadata):
    """Stack the preprocessed samples into tensors (shared by DDP processes through memory mapping)."""
    names = list(metadata.keys())
    name_to_id = {n: i for i, n in enumerate(names)}
    return {
        "images": torch.stack([s["image"] for s in data]),  # float16 tokens of a partly fine-tuned DINOv2 stay so
        "actions": torch.as_tensor(np.stack([s["actions"] for s in data]), dtype=torch.int64),
        "language_embeddings": torch.stack([s["language_embedding"] for s in data]).float(),
        "dataset_ids": torch.tensor([name_to_id[s["dataset_name"]] for s in data], dtype=torch.int64),
        "episode_indices": torch.tensor([s["episode_idx"] for s in data], dtype=torch.int64),
        "dataset_names": names,
        "dataset_name_to_id": name_to_id
    }


def _make_dataloader(dataset, indices, world_size, rank, batch_size, is_train):
    """
    Training: each process draws samples with replacement, weighted so every dataset is drawn equally often.
    Validation: each process evaluates its own shard in a fixed order.
    """
    subset = Subset(dataset, indices)
    if is_train:
        ids = dataset.dataset_ids[torch.as_tensor(indices, dtype=torch.long)]
        weights = torch.bincount(ids, minlength=len(dataset.dataset_names)).clamp_min(1)[ids].double().reciprocal()
        sampler = DistributedWeightedSampler(weights, len(indices) // world_size + 1, world_size, rank)
    else:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(subset, world_size, rank, shuffle=False)
    return DataLoader(subset, batch_size, sampler=sampler, num_workers=1,
                      worker_init_fn=worker_init_fn, persistent_workers=True)


def make_fixed_overfit_loaders(train_loader, n_samples=128, batch_size=16, seed=42):
    """
    Loaders over the same n_samples training samples, chosen randomly but fixed by seed: shuffled for training
    (every sample used once per epoch), in a fixed order for evaluation.
    Used for testing whether the model is able to memorize a small dataset.
    """
    dataset = train_loader.dataset
    indices = np.random.default_rng(seed).choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    fixed_dataset = Subset(dataset, indices.tolist())
    return (DataLoader(fixed_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(fixed_dataset, batch_size=batch_size, shuffle=False, num_workers=0))
# ─────────────────────────────────────────────────────────────────────────────
# Simulated Dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_sim_demos(hdf5_path, dataset_name="simulated"):
    """Load simulated dataset from HDF5."""
    import h5py
    data, metadata = [], {dataset_name: {}}
    total_samples = 0
    with h5py.File(hdf5_path, "r") as f:
        for ep_idx, demo_key in enumerate(sorted(f.keys())):
            g = f[demo_key]
            rgb, depth, proprio = g["rgb"][:], g["depth"][:], g["proprio"][:]
            ee_pos, ee_orn, actions = g["ee_pos"][:], g["ee_orn"][:], g["actions"][:]
            sphere_pos, n_steps = g["sphere_pos"][:], actions.shape[0]
            total_samples += n_steps
            for t in range(n_steps):
                data.append({"actions": actions[t], "image": rgb[t], "depth": depth[t],
                             "proprio": proprio[t], "ee_pos": ee_pos[t], "ee_orn": ee_orn[t],
                             "sphere_pos": sphere_pos, "episode_idx": ep_idx, "dataset_name": dataset_name})
    metadata[dataset_name] = {"image_keys": ["image"], "action_keys": ["actions"], "total_samples": total_samples}
    return data, metadata


def add_language_instruction_simulation(sim_data):
    for s in sim_data: s["language_instruction"] = "reach the green sphere"
    return sim_data


def save_sim_image_stats(sim_data):
    all_img = torch.stack([TF.to_tensor(s["image"]) for s in sim_data])
    torch.save({"mean": all_img.mean(dim=[0, 2, 3]), "std": all_img.std(dim=[0, 2, 3]).clamp(min=1e-6)},
               Path(__file__).resolve().parent.parent / "data" / "sim_image_stats.pt")


def get_action_percentiles(sim_data):
    stacked = np.stack([s["actions"] for s in sim_data])
    p1, p99 = np.percentile(stacked, [1, 99], axis=0)
    return stacked, {"simulated": {"actions": {"p1": p1, "p99": p99}}}


def get_dataloaders_sim(cfg="cfg.json", n_bins=N_BINS, keys_to_keep=None, renaming_keys=None, batch_size=16):
    """Build dataloaders for simulated dataset fine-tuning."""
    settings = get_settings(settings=str(Path(__file__).resolve().parent.parent / f"{cfg}"))
    sim_path = Path(__file__).resolve().parent.parent / "data" / settings["data_name_sim"]
    sim_data, sim_meta = load_sim_demos(sim_path)
    add_language_instruction_simulation(sim_data)
    save_sim_image_stats(sim_data)
    stacked, sim_pct = get_action_percentiles(sim_data)
    p1, p99 = sim_pct["simulated"]["actions"]["p1"], sim_pct["simulated"]["actions"]["p99"]
    sim_data = [dict(s) for s in sim_data]
    for s, d in zip(sim_data, discretize(stacked, p1, p99, n_bins).reshape(len(sim_data), -1)):
        s["actions"] = d.squeeze()
    sim_data = image_resizing(sim_data, 224)
    sim_data = filter_keys(sim_data, keys_to_keep or {}, renaming_keys={"simulated": []})
    sim_data, lang_dim, _ = add_language_embeddings(sim_data)
    sim_train, sim_val = split_by_episode(sim_data)
    return (DataLoader(Subset(sim_data, sim_train), batch_size, True),
            DataLoader(Subset(sim_data, sim_val), batch_size, False), lang_dim)