# import os
# import time

# _import_start = time.perf_counter()

# def import_log(message):
#     rank = os.environ.get("RANK", "?")
#     elapsed = time.perf_counter() - _import_start

#     print(
#         f"[Preprocessing import][Rank {rank}] "
#         f"{message} ({elapsed:.2f}s)",
#         flush=True
#     )

# import_log("Starting preprocessing imports")

# import_log("Importing torch")
import torch

# import_log("Importing torch.utils.data")
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler

# import_log("Importing preprocessing.utils")
from preprocessing.utils import get_settings

# import_log("Importing json/pathlib/numpy")
import json
from pathlib import Path
import numpy as np

# import_log("Preprocessing imports completed")

N_BINS = 32
LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "language_models" / "all-MiniLM-L6-v2"

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def open_combined_dataset(path_data, oxe_dataset_name, data_name_lst=None):
    """Load combined dataset from HDF5, returns list[dict] and metadata."""
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
                images = {k: g["images"][k][:] for k in g["images"].keys()}
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
        edges = np.linspace(p1[d], p99[d], n_bins + 1)
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


def image_resizing(data, out_dim=224, mean=None, std=None):
    from torchvision import transforms
    import torchvision.transforms.functional as TF

    """Resize and normalize images, computing stats per (dataset, episode) by default."""
    ep_to_idx = episode_to_ids(data)
    for (ds_name, ep_idx), indices in ep_to_idx.items():
        stack = torch.stack([TF.to_tensor(data[i]["image"]) for i in indices])
        ep_mean = mean if mean is not None else stack.mean(dim=[0, 2, 3])
        ep_std = std if std is not None else stack.std(dim=[0, 2, 3]).clamp(min=1e-6)
        transform = transforms.Compose([
            transforms.Resize((out_dim, out_dim), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(ep_mean, ep_std)
        ])
        for i, img in zip(indices, stack):
            data[i]["image"] = transform(img)
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
    Split episodes into train/validation sets.

    For SharedMemoryRobotDataset, use NumPy operations to avoid
    constructing large Python dictionaries and lists.

    Returns:
        train_idx, val_idx: NumPy arrays for SharedMemoryRobotDataset,
                            Python lists otherwise.
    """

    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1.")

    rng = np.random.default_rng(seed)

    if isinstance(dataset, SharedMemoryRobotDataset):

        # Access the underlying CPU tensors as NumPy views (no data copy).
        ds = dataset.dataset_ids.numpy()
        ep = dataset.episode_indices.numpy()

        if len(ds) == 0:
            raise ValueError("Cannot split an empty dataset.")

        # Encode (dataset_id, episode_id) into unique integer keys.
        # Episode IDs are nonnegative and restart for each dataset.
        base = int(ep.max()) + 1
        episode_keys = ds * base + ep

        # Identify unique episodes and map every sample to its episode.
        unique_keys, inverse = np.unique(episode_keys, return_inverse=True)

        n_episodes = len(unique_keys)

        if n_episodes < 2:
            raise ValueError(
                "At least two episodes are required for a train/validation split."
            )

        # Randomly select validation episodes.
        perm = rng.permutation(n_episodes)

        n_val = max(1, int(val_frac * n_episodes))
        n_val = min(n_val, n_episodes - 1)

        val_episodes = perm[:n_val]

        # Create a boolean lookup table for validation episodes.
        val_episode_mask = np.zeros(n_episodes, dtype=bool)
        val_episode_mask[val_episodes] = True

        # Assign every sample to train or validation.
        val_mask = val_episode_mask[inverse]

        train_idx = np.flatnonzero(~val_mask)
        val_idx = np.flatnonzero(val_mask)
        return train_idx, val_idx

    else:
        # Original implementation for regular Python datasets.
        ep_to_idx = episode_to_ids(dataset)

        ep_ids = sorted(ep_to_idx.keys())

        if len(ep_ids) < 2:
            raise ValueError(
                "At least two episodes are required for a train/validation split."
            )

        rng.shuffle(ep_ids)

        n_val = max(1, int(val_frac * len(ep_ids)))
        n_val = min(n_val, len(ep_ids) - 1)

        val_eps = set(ep_ids[:n_val])
        train_eps = ep_ids[n_val:]

        train_idx = [i for ep in train_eps for i in ep_to_idx[ep]]
        val_idx = [i for ep in val_eps for i in ep_to_idx[ep]]
        return train_idx, val_idx


# ─────────────────────────────────────────────────────────────────────────────
# Data Loaders
# ─────────────────────────────────────────────────────────────────────────────

def make_weights(subset, metadata, scale=150):
    """Compute sample weights for weighted sampling (1/n_samples per dataset)."""
    dw = {n: scale / m["total_samples"] for n, m in metadata.items()}
    return [dw[subset.dataset[i]["dataset_name"]] for i in subset.indices]


def worker_init_fn(worker_id):
    torch.set_num_threads(1)


class SharedMemoryRobotDataset(Dataset):
    """Dataset backed by shared memory tensors for DDP."""

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
                        local_rank=0, world_size=1, preprocessed_path=None):
    """Build OXE dataloaders with optional DDP and preprocessed data support."""
    import torch.distributed as dist
    settings = get_settings(settings=str(Path(__file__).resolve().parent.parent / f"{cfg}"))
    oxe_name, path_data = settings["oxe_dataset_name"], settings["path_data"]
    if data_name_lst is None: data_name_lst = settings.get("data_name_lst")
    keys_to_keep = keys_to_keep or settings["keys_to_keep"]
    renaming_keys = renaming_keys or settings["renaming_keys"]

    if use_ddp:

        rank = dist.get_rank()

        dataset_path = (
            Path(preprocessed_path)
            if preprocessed_path
            else Path("./preprocessed_data/preprocessed_dataset.pt")
        )

        # Only global rank 0 preprocesses
        if rank == 0:

            # Reuse an existing preprocessed file when explicitly provided
            use_existing = preprocessed_path is not None and dataset_path.exists()

            if not use_existing:

                data, metadata = open_combined_dataset(path_data, oxe_name, data_name_lst)

                print(f"Total samples: {len(data)}", flush=True)

                data, action_pct = apply_discretization(data, metadata, n_bins)
                data = filter_keys(data, keys_to_keep, renaming_keys)
                data = image_resizing(data)
                data, lang_dim, _ = add_language_embeddings(data)
                shared = _build_shared_data(data, metadata)
                payload = {
                    **shared,
                    "metadata": metadata,
                    "action_percentiles": action_pct,
                    "lang_dim": lang_dim
                }

                dataset_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, dataset_path)
                print(f"Preprocessed data saved to {dataset_path}", flush=True)

                # Release preprocessing memory
                import gc

                del payload
                del shared
                del data
                del metadata
                del action_pct

                gc.collect()

        # Wait until rank 0 has finished writing
        dist.barrier()

        # Every rank loads the dataset using memory mapping
        loaded = torch.load(
            str(dataset_path),
            map_location="cpu",
            mmap=True,
            weights_only=False
        )

        dataset = SharedMemoryRobotDataset(loaded)
        lang_dim = loaded["lang_dim"]
        train_idx, val_idx = split_by_episode(dataset)
        train_loader = _make_dataloader(dataset, train_idx, world_size, rank, batch_size, True)
        val_loader = _make_dataloader(dataset, val_idx, world_size, rank, batch_size, False)

        return train_loader, val_loader, lang_dim
    else:
        # Single-process mode
        data, metadata = open_combined_dataset(path_data, oxe_name, data_name_lst)
        data, _ = apply_discretization(data, metadata, n_bins)
        data = filter_keys(data, keys_to_keep, renaming_keys)
        data = image_resizing(data)
        data, lang_dim, _ = add_language_embeddings(data)
        train_idx, val_idx = split_by_episode(data)
        train_set, val_set = Subset(data, train_idx), Subset(data, val_idx)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, 
                                    sampler=WeightedRandomSampler(
                                        make_weights(train_set, metadata),
                                        num_samples=len(train_set),
                                        replacement=True
                                    ), num_workers=2, worker_init_fn=worker_init_fn
                                )
        val_loader = DataLoader(val_set, batch_size, False, num_workers=2, worker_init_fn=worker_init_fn)
        return train_loader, val_loader, lang_dim

def _build_shared_data(data, metadata):
    """Convert preprocessed data to shared memory tensors."""

    N = len(data)

    img_shape = data[0]["image"].shape
    n_actions = data[0]["actions"].shape[0]
    lang_dim = data[0]["language_embedding"].shape[0]

    names = list(metadata.keys())
    name_to_id = {n: i for i, n in enumerate(names)}

    images = torch.empty((N,) + img_shape, dtype=torch.float32)
    actions = torch.empty((N, n_actions), dtype=torch.int64)
    lang_emb = torch.empty((N, lang_dim), dtype=torch.float32)

    ds_ids = torch.empty((N,), dtype=torch.int64)
    ep_idx = torch.empty((N,), dtype=torch.int64)

    for i, s in enumerate(data):
        images[i] = s["image"]
        actions[i] = torch.from_numpy(s["actions"])
        lang_emb[i] = s["language_embedding"]

        ds_ids[i] = name_to_id[s["dataset_name"]]
        ep_idx[i] = s["episode_idx"]

    return {
        "images": images,
        "actions": actions,
        "language_embeddings": lang_emb,
        "dataset_ids": ds_ids,
        "episode_indices": ep_idx,
        "dataset_names": names,
        "dataset_name_to_id": name_to_id
    }


def _make_dataloader(dataset, indices, world_size, rank, batch_size, is_train):
    """Create dataloader with appropriate sampler for DDP or single-process."""
    subset = Subset(dataset, indices)
    if is_train:
        ids = dataset.dataset_ids[torch.as_tensor(indices, dtype=torch.long)]
        counts = torch.bincount(ids,minlength=len(dataset.dataset_names)).clamp_min(1)
        weights = counts[ids].double().reciprocal()
        sampler = DistributedWeightedSampler(weights,len(indices) // world_size + 1,world_size,rank)
        return DataLoader(subset, batch_size, sampler=sampler, num_workers=1,
                          worker_init_fn=worker_init_fn, persistent_workers=True)
    else:
        from torch.utils.data.distributed import DistributedSampler as DDistSampler
        sampler = DDistSampler(subset, world_size, rank, shuffle=False)
        return DataLoader(subset, batch_size, sampler=sampler, num_workers=1,
                          worker_init_fn=worker_init_fn, persistent_workers=True)


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