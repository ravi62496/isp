import os
import torch
import random
import numpy as np

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def setup_seed(seed=None):
    if seed is None:
        seed = random.randint(0, 10000)
        print('seed=', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
