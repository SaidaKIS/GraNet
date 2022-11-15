import numpy as np
import os
import torch
from easydict import EasyDict as edict

cfg = edict()
cfg.g.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.g.in_len = 6
cfg.g.n_hidden = 32
cfg.g.h = 64
cfg.g.w = 64
cfg.g.batch = 10


