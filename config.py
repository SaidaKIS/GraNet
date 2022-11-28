import numpy as np
import os
import torch
from easydict import EasyDict as edict

cfg = edict()
cfg.root = 'data/Masks_test/' # Raw full IMaX maps (3 for training and 1 for validate)
cfg.l = 10000 #submap dataset
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.seq_len = 5 # Length of the temporal sequence 
cfg.n_hidden = 16 # initial hidden layers
cfg.h = 128 #height of the box
cfg.w = 128 #width of the box
cfg.batch = 32 
cfg.bin_classes = ['Intergranular lane', 'Uniform-shaped granules', 'Granules with a dot', 'Granules with a lane',
                'Complex-shaped granules']
cfg.channels = 1 # initial channels
cfg.N_EPOCHS = 100 
cfg.loss = 'FocalLoss' # 'CrossEntropy', 'FocalLoss', 'mIoU'
cfg.lr = 3e-4 #Learing rate - inicial 1e-3 test 3e-4
cfg.dropout = True



