import random

import numpy as np
import torch


def fixseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


SEED = 10
EVALSEED = 0
# Provoc warning: not fully functional yet
# torch.set_deterministic(True)
torch.backends.cudnn.benchmark = False

fixseed(SEED)
