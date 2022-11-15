import numpy as np
import os
import torch
from easydict import EasyDict as edict

cfg = edict()
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.in_len = 5
cfg.n_hidden = 16
cfg.h = 128
cfg.w = 128
cfg.batch = 4


