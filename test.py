import numpy as np
import torch.backends.cudnn as cudnn

import sys
import os
import argparse
import pprint

import logging
import time
import glob
import shutil
from mmcv import Config
from mmcv.runner import init_dist, get_dist_info

import architecture_code
import models
from dataset import dataset_entry
from attack import *
import utils
import lr_scheduler

architecture = np.load('outputs/architectures.npy')

binary_architecture = np.load('outputs/binary_architectures.npy')


parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default='./experiments/RobNet_free_cifar10/config.py',
                    help='location of the config file')
#parser.add_argument('--distributed', action='store_true', default=False, help='Distributed training')
#parser.add_argument('--eval_only', action='store_true', default=False, help='Only evaluate')
parser.set_defaults(augment=True)
args = parser.parse_args()

cfg = Config.fromfile(args.config)

# Set seed
np.random.seed(cfg.seed)
cudnn.benchmark = True
torch.manual_seed(cfg.seed)
cudnn.enabled = True
torch.cuda.manual_seed(cfg.seed)

# Model
print('==> Building model..')
arch_code = eval('architecture_code.{}'.format(cfg.model))
net = models.model_entry(cfg, arch_code)
print(net.type)
print(net)