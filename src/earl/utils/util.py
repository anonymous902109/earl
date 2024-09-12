import os
import random

import torch
import numpy as np



def seed_everything(seed):
    seed_value = seed
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    g = torch.Generator()
    g.manual_seed(seed_value)

