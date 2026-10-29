import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, random_split, Subset, WeightedRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as TF
from sentence_transformers import SentenceTransformer
from preprocessing.utils import get_settings

import json
from pathlib import Path
import h5py
import numpy as np

N_BINS = 32
LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "language_models" / "all-MiniLM-L6-v2"
#################################################
################# Open dataset ##################
#################################################
def open_combined_dataset(path_data: str, oxe_dataset_name: str, data_name_lst: list = None):
    """
    Load a combined dataset from HDF5 file.

    Args:
        path_data: Path to the data directory.
        oxe_dataset_name: Name of the combined HDF5 file (without .h5 extension).
        data_name_lst: Optional list of dataset names to include. If None,
            all datasets in the HDF5 file are loaded.

    Returns:
        data: List of sample dicts.
        metadata: Dict mapping dataset_name -> metadata dict.
    """
    data = []
    metadata = {}
    with h5py.File(f"{path_data}/{oxe_dataset_name}.h5", "r") as f:
        # Filter datasets if data_name_lst is provided
        dataset_names_to_load = data_name_lst if data_name_lst is not None else f.keys()

        for data_name in dataset_names_to_load: # access a single dataset.
            if data_name not in f.keys():
                print(f"Warning: Dataset '{data_name}' not found in HDF5 file. Skipping.")
                continue

            total_samples = 0
            data_grp = f[data_name]
            metadata[data_name] = {"actions_metadata": json.loads(data_grp.attrs["actions_metadata"])}
            print(f"n_episodes: {len(data_grp.keys())}")
            for episode_idx, episode_key in enumerate(data_grp.keys()): # access a single episode.
                grp = data_grp[episode_key]
                grp_images = grp["images"]
                grp_action = grp["actions"]
                grp_language = grp["language"]
                
                # load data once per episode
                image_arrays  = {key: grp_images[key][:] for key in grp_images.keys()}
                action_arrays = {key: grp_action[key][:] for key in grp_action.keys()}
                language_arrays = {key: grp_language[key][:] for key in grp_language.keys()}

                # n_steps from any action array (they should all share dim 0)
                n_steps = next(iter(action_arrays.values())).shape[0]
                total_samples += n_steps

                # flatten episode into per-timestep dicts
                for t in range(n_steps):
                    sample = {key: arr[t][:-1] if arr.ndim == 2 else arr[t] for key, arr in action_arrays.items()}
                    sample.update({key: arr[t] for key, arr in image_arrays.items()})
                    sample.update({key: arr[t] for key, arr in language_arrays.items()})
                    sample["episode_idx"] = episode_idx
                    sample["dataset_name"] = data_name  # useful if you later mix datasets

                    data.append(sample)
                    
            metadata[data_name]["image_keys"] = list(grp_images.keys())
            metadata[data_name]["action_keys"] = list(grp_action.keys())
            metadata[data_name]["language_keys"] = list(grp_language.keys())
            metadata[data_name]["total_samples"] = total_samples
    return data, metadata


def preprocess_dataset_rank0(path_data: str, oxe_dataset_name: str, data_name_lst: list,
                              keys_to_keep: dict, renaming_keys: dict, n_bins: int = N_BINS):
    """
    Preprocess dataset on rank 0 only, then share via shared memory.

    This function performs all expensive preprocessing (loading, discretization,
    image resizing, language embeddings) once, then returns the data in a format
    suitable for shared memory sharing across DDP ranks.

    Returns:
        data: List of preprocessed sample dicts with shared memory tensors
        metadata: Dict mapping dataset_name -> metadata dict
        action_percentiles: Dict of action percentiles per dataset
        lang_dim: Dimension of language embeddings
    """
    data, metadata = open_combined_dataset(path_data, oxe_dataset_name, data_name_lst)
    data_discretized, action_percentiles = apply_discretization(data, metadata, n_bins)
    data_filtered = filter_keys(data_discretized, keys_to_keep, renaming_keys)
    data_filtered = image_resizing(data_filtered)
    data_filtered, lang_dim, _ = add_language_embeddings(data_filtered)

    return data_filtered, metadata, action_percentiles, lang_dim


def build_shared_memory_dataset(data, metadata):
    """
    Convert a preprocessed dataset into shared memory tensors for DDP.

    Args:
        data: List of preprocessed sample dicts
        metadata: Dataset metadata

    Returns:
        shared_data: Dict containing shared memory tensors:
            - images: (N, 3, H, W) float32 tensor in shared memory
            - actions: (N, action_dim) int64 tensor in shared memory
            - language_embeddings: (N, lang_dim) float32 tensor in shared memory
            - dataset_ids: (N,) int64 tensor in shared memory
            - episode_indices: (N,) int64 tensor in shared memory
            - index_to_sample: dict mapping global index to (dataset_idx, sample_idx)
        metadata: Original metadata
        action_percentiles: Action percentiles (loaded from saved file)
        lang_dim: Language embedding dimension
    """
    from collections import defaultdict
    import tempfile

    N = len(data)

    # Get dimensions
    sample = data[0]
    img_shape = sample['image'].shape  # (3, H, W)
    action_dim = sample['actions'].shape[0] if sample['actions'].ndim > 0 else 1
    lang_dim = sample['language_embedding'].shape[0]

    # Create dataset name to id mapping
    dataset_names = list(metadata.keys())
    dataset_name_to_id = {name: i for i, name in enumerate(dataset_names)}

    # Create shared memory tensors
    # Note: We need to store these in a location accessible to all processes
    # For now, use a temporary directory that can be accessed by all ranks

    # Create tensors that will hold the data
    images = torch.empty((N,) + img_shape, dtype=torch.float32)
    actions = torch.empty((N, action_dim), dtype=torch.int64)
    language_embeddings = torch.empty((N, lang_dim), dtype=torch.float32)
    dataset_ids = torch.empty((N,), dtype=torch.int64)
    episode_indices = torch.empty((N,), dtype=torch.int64)

    # Fill tensors
    for i, sample in enumerate(data):
        images[i] = sample['image']
        actions[i] = torch.from_numpy(sample['actions'])
        language_embeddings[i] = sample['language_embedding']
        dataset_ids[i] = dataset_name_to_id[sample['dataset_name']]
        episode_indices[i] = sample['episode_idx']

    # Put tensors in shared memory
    images.share_memory_()
    actions.share_memory_()
    language_embeddings.share_memory_()
    dataset_ids.share_memory_()
    episode_indices.share_memory_()

    # Create index mapping
    index_to_sample = {}
    dataset_sample_counts = defaultdict(int)
    for i, sample in enumerate(data):
        ds_name = sample['dataset_name']
        ep_idx = sample['episode_idx']
        sample_in_ep = dataset_sample_counts[(ds_name, ep_idx)]
        index_to_sample[i] = {
            'dataset_name': ds_name,
            'dataset_id': dataset_name_to_id[ds_name],
            'episode_idx': ep_idx,
            'sample_in_episode': sample_in_ep
        }
        dataset_sample_counts[(ds_name, ep_idx)] += 1

    return {
        'images': images,
        'actions': actions,
        'language_embeddings': language_embeddings,
        'dataset_ids': dataset_ids,
        'episode_indices': episode_indices,
        'index_to_sample': index_to_sample,
        'dataset_names': dataset_names,
        'dataset_name_to_id': dataset_name_to_id
    }, metadata, action_percentiles, lang_dim


def get_shared_dataset_path(shared_data_dir: str = None):
    """Get path for shared dataset storage."""
    if shared_data_dir is None:
        import tempfile
        return tempfile.mkdtemp(prefix='shared_dataset_')
    return shared_data_dir

#################################################
######### Discretize action space ###############
#################################################
def discretize(values: np.ndarray, p1: np.ndarray, p99: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """
    Maps continuous values (n_samples, action_dim) to discrete bins [0, n_bins-1],
    independently per dimension, using p1/p99 as the [lower, upper] range.
    Values outside [p1, p99] are clipped to the nearest bin.
    Compute 1st/99th percentiles
    """
    action_dim = values.shape[-1] if values.ndim > 1 else 1 # values.shape = (n_steps*n_episodes, action_dimensions).
    values = values.reshape(-1, action_dim) # reshape in case shape is (n_samples,).

    discretized = np.zeros_like(values, dtype=np.int64)

    for d in range(action_dim):
        bin_edges = np.linspace(p1[d], p99[d], n_bins + 1)
        binned = np.digitize(values[:, d], bin_edges[1:-1], right=True)
        discretized[:, d] = np.clip(binned, 0, n_bins - 1)

    return discretized.squeeze()

def undiscretize(bin_indices, p1, p99, n_bins: int = N_BINS) -> torch.Tensor:
    """
    Inverse of discretize(): maps bin indices [0, n_bins-1] back to continuous
    values, using the midpoint of each bin's range as the reconstruction.

    Accepts bin_indices/p1/p99 as either numpy arrays or torch tensors --
    normalizes everything to torch internally, so mixed numpy/torch inputs
    (a common source of confusing errors) can never reach the arithmetic below.

    bin_indices: (B, action_dim) int
    p1, p99:     (action_dim,) float -- same percentiles used during discretization
    Returns:     (B, action_dim) float torch.Tensor -- reconstructed continuous actions
    """
    if isinstance(bin_indices, np.ndarray):
        bin_indices = torch.from_numpy(bin_indices)
    if isinstance(p1, np.ndarray):
        p1 = torch.from_numpy(p1)
    if isinstance(p99, np.ndarray):
        p99 = torch.from_numpy(p99)

    bin_indices = bin_indices.float()
    p1 = p1.float()
    p99 = p99.float()

    bin_width = (p99 - p1) / n_bins
    continuous = p1 + (bin_indices + 0.5) * bin_width
    return continuous

def apply_discretization(data: list, metadata: dict, n_bins: int = N_BINS):
    action_percentiles = {} # keep track of action percentiles per dataset per action dimension.
    data_discretized = []

    for data_name in list(metadata.keys()):#metadata.keys():
        action_keys = metadata[data_name]["action_keys"]

        # filter samples belonging to this dataset only
        batch = [sample for sample in data if sample["dataset_name"] == data_name]
        
        action_percentiles[data_name] = {}

        # start each new sample as a shallow copy of the original (keeps images, episode_idx, dataset_name, etc.)
        batch_discretized = [dict(sample) for sample in batch]

        for action_key in action_keys:
            stacked = np.stack([sample[action_key] for sample in batch])
            if stacked.ndim == 1:
                stacked = stacked.reshape(-1, 1)

            p1 = np.percentile(stacked, 1, axis=0)
            p99 = np.percentile(stacked, 99, axis=0)
            action_percentiles[data_name][action_key] = {"p1": p1, "p99": p99}

            discretized = discretize(stacked, p1, p99, n_bins)

            # overwrite the continuous action_key with its discretized version
            for sample_disc, disc_val in zip(batch_discretized, discretized.reshape(len(batch), -1)):
                sample_disc[action_key] = disc_val.squeeze()

            print(f"{data_name}\n{action_key}: shape={stacked.shape}\n"
                f"p1={p1}\np99={p99}\ndiscretized_range=[{discretized.min()}, {discretized.max()}]\n\n")

        data_discretized.extend(batch_discretized)
    return data_discretized, action_percentiles

#################################################
############ Filter & Rename keys  #############
#################################################
def filter_keys(
    data: list[dict],
    keys_to_keep: dict[str, list[str]],
    renaming_keys: dict[str, list[list[str, str]]] = None,
) -> list[dict]:
    """
    Filters each sample in `data` down to a subset of keys, chosen per-dataset,
    and optionally renames some of those keys per-dataset.

    Args:
        data: list of sample dicts, each with a "dataset_name" field.
        keys_to_keep: maps dataset_name -> list of keys to preserve for
            samples belonging to that dataset (using the ORIGINAL key names).
        renaming_keys: maps dataset_name -> list of (old_name, new_name) tuples.
            Applied AFTER filtering, so old_name must be one of the kept keys.
            e.g.:
            {
                "taco_play": [("rgb_static", "image")],
            }

    Returns:
        A new list of filtered (and possibly renamed) dicts. Original `data`
        is left untouched.
    """
    renaming_keys = renaming_keys or {}
    filtered = []

    for sample in data:
        dataset_name = sample["dataset_name"]
        keys = keys_to_keep[dataset_name]  # which keys to keep for this sample's dataset

        filtered_sample = {key: sample[key] for key in keys}

        # apply renaming, if any is defined for this dataset
        for old_name, new_name in renaming_keys.get(dataset_name, []):
            filtered_sample[new_name] = filtered_sample.pop(old_name)

        filtered.append(filtered_sample)

    return filtered




#################################################
################ Resize Images  #################
#################################################
def episode_to_ids(dataset: list[dict]):
  # key: (dataset_name, episode_idx); value: list of sample indices belonging to that episode
  episode_to_indices: dict[tuple[str, int], list[int]] = {}
  for i, sample in enumerate(dataset):
      key = (sample["dataset_name"], sample["episode_idx"])
      episode_to_indices.setdefault(key, []).append(i)
  return episode_to_indices

def image_resizing(data: list[dict], out_image_dim: int = 224, mean: torch.Tensor = None, std: torch.Tensor = None):
    """
    Resize + normalize images in `data`.

    If mean/std are not provided, they are computed separately for each
    (dataset_name, episode_idx, image_key) group — since lighting/scene
    statistics can vary a lot across episodes and datasets, normalizing
    globally would blur that out.

    If mean/std ARE provided, that FIXED pair is used for every image instead
    of computing per-episode stats -- useful for deployment/inference time,
    where you don't have a whole episode available in advance to compute
    statistics from (e.g. a live closed-loop rollout, one frame at a time).

    Modifies `data` in place: image arrays are replaced with normalized,
    resized torch.Tensor images of shape (3, out_image_dim, out_image_dim).
    """
    # group sample indices by (dataset_name, episode_idx)
    episode_to_indices: dict[tuple[str, int], list[int]] = {}
    for i, sample in enumerate(data):
        key = (sample["dataset_name"], sample["episode_idx"])
        episode_to_indices.setdefault(key, []).append(i)

    for (dataset_name, episode_idx), indices in episode_to_indices.items():
        # gather every image for this image_key within this one episode
        raw_images = [data[i]['image'] for i in indices]  # each (H, W, 3) uint8

        # convert to tensor stack (N, 3, H, W), scaled to [0, 1]
        episode_stack = torch.stack([TF.to_tensor(img) for img in raw_images])

        # use provided mean/std if given, otherwise compute per-episode stats
        if mean is None:
            episode_mean = episode_stack.mean(dim=[0, 2, 3])
        else:
            episode_mean = mean

        if std is None:
            episode_std = episode_stack.std(dim=[0, 2, 3]).clamp(min=1e-6)  # avoid div-by-zero
        else:
            episode_std = std

        
        transform = transforms.Compose([
            transforms.Resize((out_image_dim, out_image_dim),
                                interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=episode_mean, std=episode_std),
        ])

        # apply resize+normalize to each image, write back into data in place
        for i, img_tensor in zip(indices, episode_stack):
            data[i]['image'] = transform(img_tensor)

    return data



#################################################
##### Embedding Language Instructions ###########
#################################################
def _to_str(x) -> str:
    if isinstance(x, np.ndarray):
        x = x.item()
    return x.decode("utf-8") if isinstance(x, bytes) else str(x)

def load_text_model(model_name="all-MiniLM-L6-v2", local_dir=LOCAL_MODEL_DIR):
    """Load from a local folder if present; otherwise download once and save it there."""
    if (local_dir / "modules.json").exists():
        return SentenceTransformer(str(local_dir))          # no network access
    model = SentenceTransformer(model_name)                  # one-time download
    local_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(local_dir))
    return model

def add_language_embeddings(data,
                            model_name="all-MiniLM-L6-v2", out_key="language_embedding"):
    """
    instruction_keys: dataset_name -> which entry of metadata[...]["language_keys"]
    holds the natural-language instruction, e.g.
      {"taco_play": "natural_language_instruction", "stanford_hydra...": "..."}
    Encodes each UNIQUE instruction once (frozen encoder) and stores a (D_text,)
    float tensor under sample[out_key].
    """
    texts = [_to_str(s['language_instruction']) for s in data]
    uniq = sorted(set(texts))
    print(f"{len(uniq)} unique instructions")

    text_model = load_text_model(model_name)
    emb = text_model.encode(uniq, convert_to_tensor=True, normalize_embeddings=True).cpu()
    lookup = {t: e for t, e in zip(uniq, emb)}

    for s, t in zip(data, texts):
        s[out_key] = lookup[t].clone()
    return data, emb.shape[1], text_model   # keep text_model for rollout-time encoding


#################################################
############### Build Data Loaders ##############
#################################################
def split_by_episode(dataset: list[dict] | SharedMemoryRobotDataset, val_frac: float = 0.1, seed: int = 42):
    '''
    - For each sample, get its (dataset_name, episode_idx) pair, since episode_idx
      alone is not globally unique (it resets to 0 for each dataset_name).
    - Split episodes into train/val sets, constrained to not overlap.
    - Use sample indices to build train/val Subsets, enforcing different episodes
      (within each dataset_name) across the two splits.
    - Episodes appearing in training set must not appear in validation set.
    '''

    # Handle both regular list[dict] and SharedMemoryRobotDataset
    if isinstance(dataset, SharedMemoryRobotDataset):
        # For shared memory dataset, use the index_to_sample mapping
        episode_to_indices: dict[tuple[str, int], list[int]] = {}
        for i in range(len(dataset.images)):
            sample_info = dataset.index_to_sample[i]
            key = (sample_info['dataset_name'], sample_info['episode_idx'])
            episode_to_indices.setdefault(key, []).append(i)
        all_indices = list(range(len(dataset.images)))
    else:
        # Original behavior for list[dict]
        episode_to_indices = episode_to_ids(dataset)
        all_indices = list(range(len(dataset)))

    episode_ids = sorted(episode_to_indices.keys())  # sorted list of (dataset_name, episode_idx) tuples

    rng = np.random.default_rng(seed)
    rng.shuffle(episode_ids)

    # select different episodes for train vs validation
    n_val_ep = max(1, int(val_frac * len(episode_ids)))
    train_episodes = set(episode_ids[n_val_ep:]) # keys of episode_to_indices going to training data.
    val_episodes = set(episode_ids[:n_val_ep])

    # sample indices, i.e., indices of dataset[idx]
    train_indices = [i for ep in train_episodes for i in episode_to_indices[ep]]
    val_indices   = [i for ep in val_episodes   for i in episode_to_indices[ep]]

    return train_indices, val_indices  # Return indices only, let caller build subsets

def make_weights(subset, metadata):
  '''
  List of weights with length equal to subset length.
  Individual datasets stored within subset are not sorted.
  Loop over subset and save the weight associated with the
  corresponding dataset.

  Weights are computed as 1/n_samples per dataset, scaled by a factor
  to make weights more interpretable.
  '''

  scaling_factor = 150 # to make weights not too small.

  # Compute weight for each dataset dynamically based on total samples
  dataset_weight = {}
  for data_name, meta in metadata.items():
    n_samples = meta["total_samples"]
    dataset_weight[data_name] = 1/n_samples * scaling_factor

  return [dataset_weight[subset.dataset[i]["dataset_name"]] for i in subset.indices]

def worker_init_fn(worker_id):
    # each worker process should use just 1 thread, since it only does
    # cheap list-indexing/dict-copies here, not real tensor math
    torch.set_num_threads(1)


class SharedMemoryRobotDataset(Dataset):
    """
    Dataset backed by shared memory tensors for DDP.

    Created by rank 0 during preprocessing, then shared with all ranks.
    Each rank can access the same underlying tensor storage without copying.
    """

    def __init__(self, shared_data: dict, indices: list = None):
        """
        Args:
            shared_data: Dict containing shared memory tensors from build_shared_memory_dataset
            indices: Optional list of indices to include (for train/val split)
        """
        self.images = shared_data['images']
        self.actions = shared_data['actions']
        self.language_embeddings = shared_data['language_embeddings']
        self.dataset_ids = shared_data['dataset_ids']
        self.episode_indices = shared_data['episode_indices']
        self.index_to_sample = shared_data['index_to_sample']
        self.dataset_names = shared_data['dataset_names']
        self.dataset_name_to_id = shared_data['dataset_name_to_id']

        self.indices = indices if indices is not None else list(range(len(self.images)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map global index to actual data
        actual_idx = self.indices[idx]
        sample_info = self.index_to_sample[actual_idx]

        return {
            'image': self.images[actual_idx],
            'actions': self.actions[actual_idx],
            'language_embedding': self.language_embeddings[actual_idx],
            'dataset_id': self.dataset_ids[actual_idx],
            'episode_idx': self.episode_indices[actual_idx],
            'dataset_name': self.dataset_names[self.dataset_ids[actual_idx].item()],
        }


class DistributedWeightedSampler:
    """
    Distributed sampler with weighted sampling.

    Each rank gets a different weighted random sample of the data,
    ensuring no overlap between ranks while respecting the weighting.

    Args:
        weights: List of weights for each sample in the full dataset
        num_samples: Total number of samples to draw per epoch (per rank)
        num_replicas: Number of DDP replicas
        rank: Current replica rank
        replacement: Whether to sample with replacement
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle the sampled indices
    """

    def __init__(self, weights: list, num_samples: int, num_replicas: int, rank: int,
                 replacement: bool = True, seed: int = 42, shuffle: bool = True):
        self.weights = torch.tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

        # Total number of samples in the full dataset
        self.num_samples_full = len(self.weights)

        # Calculate roughly how many samples per rank
        self.samples_per_rank = self.num_samples_full // num_replicas
        if rank < (self.num_samples_full % num_replicas):
            self.samples_per_rank += 1

    def __iter__(self):
        # Set seed based on epoch for reproducibility
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch + self.rank * 1000)  # Add rank offset

        # Generate weighted random sample for this rank
        if self.replacement:
            # Sample with replacement from the full distribution
            # Each rank samples independently with its own seed offset
            indices = torch.multinomial(
                self.weights, self.num_samples, replacement=True, generator=g
            ).tolist()
        else:
            # Sample without replacement - shuffle by weight
            shuffled_weights = self.weights * torch.rand(self.num_samples_full, generator=g)
            _, indices = torch.topk(shuffled_weights, min(self.num_samples, self.num_samples_full))
            indices = indices.tolist()

        # Shuffle indices within this rank's sample
        if self.shuffle:
            perm = torch.randperm(len(indices), generator=g)
            indices = [indices[i] for i in perm]

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Set epoch for shuffling consistency across epochs."""
        self.epoch = epoch


def get_dataloaders_oxe(cfg: str = "cfg.json",
                        n_bins: int = N_BINS,
                        keys_to_keep: dict = None,
                        renaming_keys: dict = None,
                        batch_size: int = 32,
                        data_name_lst: list = None,
                        use_ddp: bool = False,
                        local_rank: int = 0,
                        world_size: int = 1,
                        preprocessed_path: str = None):
    """
    Build train and validation dataloaders for OxE dataset.

    In DDP mode with preprocessed_path: Load preprocessed shared-memory data saved by rank 0.
    In DDP mode without preprocessed_path: Rank 0 does preprocessing, shares via shared memory.
    In single-process mode: data is preprocessed in the usual way.

    Args:
        cfg: Path to config file
        n_bins: Number of bins for discretization
        keys_to_keep: Dict mapping dataset names to keys to keep
        renaming_keys: Dict mapping dataset names to (old, new) key tuples
        batch_size: Batch size for dataloaders
        data_name_lst: Optional list of dataset names to load
        use_ddp: Whether to use DDP mode with shared memory
        local_rank: Current process rank (for DDP)
        world_size: Total number of processes (for DDP)
        preprocessed_path: Optional path to preprocessed dataset .pt file.
                          If provided, skips expensive preprocessing.

    Returns:
        train_loader, val_loader, lang_dim
    """
    import torch.distributed as dist

    settings = get_settings(settings=str(Path(__file__).resolve().parent.parent / f"{cfg}"))

    oxe_dataset_name = settings["oxe_dataset_name"]
    path_data = settings["path_data"]

    # If data_name_lst not provided, use the one from config
    if data_name_lst is None:
        data_name_lst = settings.get("data_name_lst", None)

    if use_ddp:
        # DDP mode: Rank 0 does preprocessing, shares via shared memory
        if local_rank == 0:
            if preprocessed_path and Path(preprocessed_path).exists():
                # Load preprocessed data instead of doing expensive preprocessing
                print(f"Loading preprocessed data from {preprocessed_path}")
                loaded = torch.load(preprocessed_path, map_location='cpu')
                shared_data = {
                    'images': loaded['images'],
                    'actions': loaded['actions'],
                    'language_embeddings': loaded['language_embeddings'],
                    'dataset_ids': loaded['dataset_ids'],
                    'episode_indices': loaded['episode_indices'],
                    'index_to_sample': loaded['index_to_sample'],
                    'dataset_names': loaded['dataset_names'],
                    'dataset_name_to_id': loaded['dataset_name_to_id'],
                }
                metadata = loaded['metadata']
                action_percentiles = loaded['action_percentiles']
                lang_dim = loaded.get('lang_dim', 384)
            else:
                # Preprocess data on rank 0
                print("Preprocessing data on rank 0...")
                data, metadata = open_combined_dataset(path_data, oxe_dataset_name, data_name_lst)
                data_discretized, action_percentiles = apply_discretization(data, metadata, n_bins)
                data_filtered = filter_keys(data_discretized, keys_to_keep, renaming_keys)
                data_filtered = image_resizing(data_filtered)
                data_filtered, lang_dim, text_model = add_language_embeddings(data_filtered)

                # Build shared memory dataset
                shared_data, metadata, action_percentiles, lang_dim = build_shared_memory_dataset(
                    data_filtered, metadata
                )

                # Save preprocessed data for future runs
                save_dir = Path("./preprocessed_data")
                save_dir.mkdir(exist_ok=True)
                save_path = save_dir / "preprocessed_dataset.pt"
                torch.save({
                    'images': shared_data['images'],
                    'actions': shared_data['actions'],
                    'language_embeddings': shared_data['language_embeddings'],
                    'dataset_ids': shared_data['dataset_ids'],
                    'episode_indices': shared_data['episode_indices'],
                    'index_to_sample': shared_data['index_to_sample'],
                    'dataset_names': shared_data['dataset_names'],
                    'dataset_name_to_id': shared_data['dataset_name_to_id'],
                    'metadata': metadata,
                    'action_percentiles': action_percentiles,
                    'lang_dim': lang_dim,
                }, save_path)
                print(f"Preprocessed data saved to {save_path}")

            # Broadcast lang_dim to other ranks
            dist.broadcast(torch.tensor([lang_dim]), src=0)

            # Notify other ranks that data is ready
            dist.barrier()

            # Create the dataset from shared data (for rank 0)
            dataset = SharedMemoryRobotDataset(shared_data)

            # Build train/val indices for the full dataset
            train_indices, val_indices = split_by_episode(dataset, val_frac=0.1, seed=42)

            # Compute weights based on dataset distribution (for the full dataset)
            weights = []
            dataset_id_to_name = {v: k for k, v in shared_data['dataset_name_to_id'].items()}
            for i in range(len(dataset.images)):
                ds_id = dataset.dataset_ids[i].item()
                ds_name = dataset_id_to_name[ds_id]
                ds_samples = sum(1 for j in range(len(dataset.images))
                               if dataset.dataset_ids[j].item() == ds_id)
                weights.append(1.0 / ds_samples)

            # Create train dataset with distributed weighted sampler
            train_dataset = Subset(dataset, train_indices)

            # Compute weights for train dataset only
            train_weights = [weights[i] for i in train_indices]
            num_train_samples = len(train_indices)
            num_replicas = world_size
            rank = local_rank

            # Create distributed weighted sampler for training
            train_sampler = DistributedWeightedSampler(
                weights=train_weights,
                num_samples=num_train_samples // num_replicas + 1,  # Roughly equal per rank
                num_replicas=num_replicas,
                rank=rank,
                replacement=True,
                seed=42,
                shuffle=True
            )

            train_loader = DataLoader(
                train_dataset,
                sampler=train_sampler,
                batch_size=batch_size,
                num_workers=1,
                persistent_workers=True,
                worker_init_fn=worker_init_fn,
            )

            # Create validation dataset - each rank uses DistributedSampler to get its portion
            val_dataset = Subset(dataset, val_indices)
            num_val_samples = len(val_indices)

            # Use DistributedSampler for validation to evenly distribute samples
            from torch.utils.data.distributed import DistributedSampler as ValDistributedSampler
            val_sampler = ValDistributedSampler(
                val_dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False
            )

            val_loader = DataLoader(
                val_dataset,
                sampler=val_sampler,
                batch_size=batch_size,
                shuffle=False,
                num_workers=1,
                persistent_workers=True,
                worker_init_fn=worker_init_fn,
            )

            print(f"Rank {local_rank}: Built dataloaders, train_samples={len(train_sampler)}, val_samples={len(val_sampler)}")

            return train_loader, val_loader, lang_dim

        else:
            # Non-rank-0 processes: wait for rank 0 to finish preprocessing
            dist.barrier()

            # Load preprocessed data from shared location
            if preprocessed_path and Path(preprocessed_path).exists():
                print(f"Loading preprocessed data from {preprocessed_path}")
                loaded = torch.load(preprocessed_path, map_location='cpu')
                shared_data = {
                    'images': loaded['images'],
                    'actions': loaded['actions'],
                    'language_embeddings': loaded['language_embeddings'],
                    'dataset_ids': loaded['dataset_ids'],
                    'episode_indices': loaded['episode_indices'],
                    'index_to_sample': loaded['index_to_sample'],
                    'dataset_names': loaded['dataset_names'],
                    'dataset_name_to_id': loaded['dataset_name_to_id'],
                }
                metadata = loaded['metadata']
                action_percentiles = loaded['action_percentiles']
                lang_dim = loaded.get('lang_dim', 384)
            else:
                # Load from default preprocessed data location
                default_path = Path("./preprocessed_data/preprocessed_dataset.pt")
                if default_path.exists():
                    print(f"Loading preprocessed data from {default_path}")
                    loaded = torch.load(default_path, map_location='cpu')
                    shared_data = {
                        'images': loaded['images'],
                        'actions': loaded['actions'],
                        'language_embeddings': loaded['language_embeddings'],
                        'dataset_ids': loaded['dataset_ids'],
                        'episode_indices': loaded['episode_indices'],
                        'index_to_sample': loaded['index_to_sample'],
                        'dataset_names': loaded['dataset_names'],
                        'dataset_name_to_id': loaded['dataset_name_to_id'],
                    }
                    metadata = loaded['metadata']
                    action_percentiles = loaded['action_percentiles']
                    lang_dim = loaded.get('lang_dim', 384)
                else:
                    raise RuntimeError("Preprocessed data not found. Run rank 0 preprocessing first.")

            # Broadcast lang_dim from rank 0
            lang_dim_tensor = torch.tensor([0])
            dist.broadcast(lang_dim_tensor, src=0)
            lang_dim = lang_dim_tensor.item()

            # Build dataset from loaded shared data
            dataset = SharedMemoryRobotDataset(shared_data)

            # Build train/val indices for the full dataset
            train_indices, val_indices = split_by_episode(dataset, val_frac=0.1, seed=42)

            # Compute weights based on dataset distribution (for the full dataset)
            weights = []
            dataset_id_to_name = {v: k for k, v in shared_data['dataset_name_to_id'].items()}
            for i in range(len(dataset.images)):
                ds_id = dataset.dataset_ids[i].item()
                ds_name = dataset_id_to_name[ds_id]
                ds_samples = sum(1 for j in range(len(dataset.images))
                               if dataset.dataset_ids[j].item() == ds_id)
                weights.append(1.0 / ds_samples)

            # Create train dataset with distributed weighted sampler
            train_dataset = Subset(dataset, train_indices)

            # Compute weights for train dataset only
            train_weights = [weights[i] for i in train_indices]
            num_train_samples = len(train_indices)
            num_replicas = world_size
            rank = local_rank

            # Create distributed weighted sampler for training
            train_sampler = DistributedWeightedSampler(
                weights=train_weights,
                num_samples=num_train_samples // num_replicas + 1,
                num_replicas=num_replicas,
                rank=rank,
                replacement=True,
                seed=42,
                shuffle=True
            )

            train_loader = DataLoader(
                train_dataset,
                sampler=train_sampler,
                batch_size=batch_size,
                num_workers=1,
                persistent_workers=True,
                worker_init_fn=worker_init_fn,
            )

            # Create validation dataset
            val_dataset = Subset(dataset, val_indices)
            num_val_samples = len(val_indices)

            # Use DistributedSampler for validation
            from torch.utils.data.distributed import DistributedSampler as ValDistributedSampler
            val_sampler = ValDistributedSampler(
                val_dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False
            )

            val_loader = DataLoader(
                val_dataset,
                sampler=val_sampler,
                batch_size=batch_size,
                shuffle=False,
                num_workers=1,
                persistent_workers=True,
                worker_init_fn=worker_init_fn,
            )

            print(f"Rank {local_rank}: Built dataloaders from preprocessed data, train_samples={len(train_sampler)}, val_samples={len(val_sampler)}")

            return train_loader, val_loader, lang_dim

    else:
        # Single-process mode: original behavior
        ############## open data
        data, metadata = open_combined_dataset(path_data, oxe_dataset_name, data_name_lst)
        ############## discretize data
        data_discretized, action_percentiles = apply_discretization(data, metadata, N_BINS)

        ############## filter data
        data_filtered = filter_keys(data_discretized, keys_to_keep, renaming_keys)

        ############## resize images
        # print(data_filtered[0]['image'].shape)
        image_resizing(data_filtered)
        # print(data_filtered[0]['image'].shape)

        ############## embedding language
        data_filtered, lang_dim, text_model = add_language_embeddings(data_filtered)

        ############## Build dataloaders.
        # get training and validation subsets. Force episodes to appear in train or val, but not both.
        train_set, val_set = split_by_episode(data_filtered, val_frac=0.1, seed=42)

        train_weights = make_weights(train_set, metadata)
        val_weights   = make_weights(val_set, metadata)

        train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
        train_loader  = DataLoader(
            train_set,
            sampler=train_sampler,
            batch_size=batch_size,
            num_workers=2,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
        )

        # no weighted sampling needed for validation. Just shuffle=False, full pass
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
        )

        return train_loader, val_loader, lang_dim



#################################################
############### Simulated Dataset ###############
#################################################
def load_sim_demos(hdf5_path: str, dataset_name: str = "simulated"):
    """
    Mirrors the structure of your original HDF5-loading code, but adapted to
    collect_demos'/_save_hdf5's schema: one group per demo (episode), with
    datasets "rgb", "depth", "proprio", "ee_pos", "ee_orn", "actions",
    "sphere_pos" directly inside each group (no separate "images"/"actions"
    subgroups like the real datasets had).
    """
    data = []
    metadata = {dataset_name: {}}
    total_samples = 0

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(f.keys())  # "demo_0000", "demo_0001", ...
        print(f"n_episodes: {len(demo_keys)}")

        for episode_idx, demo_key in enumerate(demo_keys):
            grp = f[demo_key]

            # load every array for this episode once (matches your original pattern)
            rgb = grp["rgb"][:]
            depth = grp["depth"][:]
            proprio = grp["proprio"][:]
            ee_pos = grp["ee_pos"][:]
            ee_orn = grp["ee_orn"][:]
            actions = grp["actions"][:]
            # sphere_pos is per-episode (fixed target), not per-timestep -- handled separately below
            sphere_pos = grp["sphere_pos"][:]

            n_steps = actions.shape[0]
            total_samples += n_steps

            for t in range(n_steps):
                sample = {
                    "actions": actions[t],
                    "image": rgb[t],          # renamed to "image" to match your other datasets' convention
                    "depth": depth[t],
                    "proprio": proprio[t],
                    "ee_pos": ee_pos[t],
                    "ee_orn": ee_orn[t],
                    "sphere_pos": sphere_pos,   # same target for every timestep in this episode
                    "episode_idx": episode_idx,
                    "dataset_name": dataset_name,
                }
                data.append(sample)

    metadata[dataset_name]["image_keys"] = ["image"]
    metadata[dataset_name]["action_keys"] = ["actions"]
    metadata[dataset_name]["total_samples"] = total_samples

    return data, metadata

def add_language_instruction_simulation(sim_data: list):
    for idx, sim_data_element in enumerate(sim_data):
        sim_data[idx]['language_instruction'] = "reach the green sphere"
    return sim_data

def save_sim_image_stats(sim_data: list):
    '''
    Save the mean and std of the images, as they will be used when rolling out the policy.
    When doing rollout the model is used on each iteration, so there isn't data to compute
    the mean and std, thus, preprocess the images before entering the model.
    Then image_resizing (where images are normalized) takes fixed_mean and fixed_std as inputs.


    Get mean and std of images from raw data. Use for normalizing RGB (images) generated during rollout closed-loop.\
Note: Mean and std must come from the raw data, because there aren't enough episodes (only a single episode is available) when doing rollout to get the statistics.
    '''
    # compute a single fixed mean/std from all sim training images, once, after training data is built
    all_sim_images = torch.stack([TF.to_tensor(s["image"]) for s in sim_data])  # from the RAW (pre-resizing) sim_data
    fixed_mean = all_sim_images.mean(dim=[0, 2, 3])
    fixed_std = all_sim_images.std(dim=[0, 2, 3]).clamp(min=1e-6)

    # save these alongside your checkpoint/action_percentiles for later reuse at inference time
    torch.save({"mean": fixed_mean, "std": fixed_std}, str(Path(__file__).resolve().parent.parent) + "/data/sim_image_stats.pt")

def get_action_percentiles(sim_data):
    sim_action_percentiles = {}
    stacked = np.stack([s["actions"] for s in sim_data])
    p1 = np.percentile(stacked, 1, axis=0)
    p99 = np.percentile(stacked, 99, axis=0)
    sim_action_percentiles["simulated"] = {"actions": {"p1": p1, "p99": p99}}

    return stacked, sim_action_percentiles
def get_dataloaders_sim(cfg: str = "cfg.json", 
                        n_bins: int = N_BINS,
                        keys_to_keep: dict = None,
                        renaming_keys: dict = None,
                        batch_size: int = 16):


    settings = get_settings(settings=str(Path(__file__).resolve().parent.parent / f"{cfg}"))

    data_name_sim = settings["data_name_sim"]

    # 2. load + flatten
    sim_data, sim_metadata = load_sim_demos(Path(__file__).resolve().parent.parent / "data" / f"{data_name_sim}")
    add_language_instruction_simulation(sim_data)
    save_sim_image_stats(sim_data)
    # metadata.update(sim_metadata)

    # 3. discretize (own p1/p99, own action_dim=3)
    stacked, sim_action_percentiles = get_action_percentiles(sim_data)
    p1, p99 = sim_action_percentiles["simulated"]["actions"]["p1"], sim_action_percentiles["simulated"]["actions"]["p99"]

    sim_data_discretized = [dict(s) for s in sim_data]
    discretized = discretize(stacked, p1, p99, N_BINS)
    for sample, disc_val in zip(sim_data_discretized, discretized.reshape(len(sim_data), -1)):
        sample["actions"] = disc_val.squeeze()

    # action_percentiles.update(sim_action_percentiles)

    # 4. resize + normalize images (per-episode stats, since mean/std are not passed here)
    sim_data_discretized = image_resizing(sim_data_discretized, out_image_dim=224)

    # 5. trim to consistent schema
    sim_data_filtered = filter_keys(sim_data_discretized, keys_to_keep, renaming_keys={"simulated": []})

    # 6. embed language instruction
    sim_data_filtered, lang_dim, text_model_sim = add_language_embeddings(sim_data_filtered)

    # 7. split by episode, build DataLoader for fine-tuning
    sim_train_set, sim_val_set = split_by_episode(sim_data_filtered, val_frac=0.1, seed=42)
    sim_train_loader = DataLoader(sim_train_set, batch_size=batch_size, shuffle=True)
    sim_val_loader = DataLoader(sim_val_set, batch_size=batch_size, shuffle=False)

    return sim_train_loader, sim_val_loader, lang_dim


'''
============================== OxE DATASET ==============================
keys_to_keep = {
    "stanford_hydra_dataset_converted_externally_to_rlds": ["actions", "image", "language_instruction","episode_idx", "dataset_name"],
    "taco_play": ["rel_actions_world", "rgb_static", "natural_language_instruction", "episode_idx", "dataset_name"],
}

renaming_keys = {
    "taco_play": [("rgb_static", "image"), ("rel_actions_world", "actions"), ("natural_language_instruction","language_instruction")],  # so both datasets share the key "image"
}

train_loader, val_loader, lang_dim = get_dataloaders_oxe(cfg = "cfg.json", 
                    n_bins = N_BINS,
                    keys_to_keep = keys_to_keep,
                    renaming_keys = renaming_keys,
                    batch_size = 32)

batch = next(iter(train_loader))


============================== SIMULATION ==============================
'''