import os
import random
from typing import Optional

import numpy as np
import torch

max_seed_value = 4294967295  # 2^32 - 1 (uint32)
min_seed_value = 0

from dataclasses import dataclass
from typing import Optional

@dataclass
class GLiNERConfigArgs:
    # Extra arguments
    log_dir: str = "models/"

    # Model Configuration
    model_name: str = "answerdotai/ModernBERT-base"
    labels_encoder: str = "BAAI/bge-small-en-v1.5"
    name: str = "span level gliner"
    max_width: int = 12
    hidden_size: int = 768
    dropout: float = 0.3
    fine_tune: bool = True
    subtoken_pooling: str = "first"
    fuse_layers: bool = False
    post_fusion_schema: str = "l2l-l2t-t2t"
    span_mode: str = "markerV0"
    
    # Training Parameters
    num_steps: int = 100000
    num_train_epochs: int = 100000
    train_batch_size: int = 8
    eval_every: int = 5000
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    warmup_ratio: float = 0.05
    scheduler_type: str = "cosine"
    
    # Loss Function Parameters
    loss_alpha: float = 0.75
    loss_gamma: int = 0
    label_smoothing: int = 0
    loss_reduction: str = "sum"
    
    # Learning Rate and Weight Decay Configuration
    lr_encoder: float = 1e-5
    lr_others: float = 3e-5
    weight_decay_encoder: float = 0.1
    weight_decay_other: float = 0.01
    max_grad_norm: float = 10.0
    
    # Directory Paths
    root_dir: str = "gliner_logs"
    train_data: str = "data/data/train.json"
    val_data_dir: str = "data/data/test.json"
    
    # Pretrained Model Configuration
    prev_path: Optional[str] = None
    save_total_limit: int = 3
    
    # Advanced Training Settings
    size_sup: int = -1
    max_types: int = 100
    shuffle_types: bool = True
    random_drop: bool = True
    max_neg_type_ratio: int = 1
    max_len: int = 512


def seed_everything(
    seed: Optional[int] = None, workers: bool = False, verbose: bool = True
) -> int:
    r"""Function that sets the seed for pseudo-random number generators in: torch, numpy, and Python's random module.
    In addition, sets the following environment variables:

    - ``PL_GLOBAL_SEED``: will be passed to spawned subprocesses (e.g. ddp_spawn backend).
    - ``PL_SEED_WORKERS``: (optional) is set to 1 if ``workers=True``.

    Args:
            seed: the integer value seed for global random state in Lightning.
                    If ``None``, it will read the seed from ``PL_GLOBAL_SEED`` env variable. If ``None`` and the
                    ``PL_GLOBAL_SEED`` env variable is not set, then the seed defaults to 0.
            workers: if set to ``True``, will properly configure all dataloaders passed to the
                    Trainer with a ``worker_init_fn``. If the user already provides such a function
                    for their dataloaders, setting this argument will have no influence. See also:
                    :func:`~lightning_fabric.utilities.seed.pl_worker_init_function`.
            verbose: Whether to print a message on each rank with the seed being set.

    """
    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = 0
            # rank_zero_warn(f"No seed found, seed set to {seed}")
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = 0
                # rank_zero_warn(f"Invalid seed found: {repr(env_seed)}, seed set to {seed}")
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        # rank_zero_warn(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = 0

    # if verbose:
    # log.info(rank_prefixed_message(f"Seed set to {seed}", _get_rank()))

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed