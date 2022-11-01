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
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default='./experiments/RobNet_free_cifar10/config.py',
                    help='location of the config file')
parser.add_argument('--distributed', action='store_true', default=False, help='Distributed training')
parser.add_argument('--eval_only', action='store_true', default=False, help='Only evaluate')
parser.set_defaults(augment=True)
args = parser.parse_args()

archs_path = 'outputs/architectures.npy'
bin_archs_path = 'outputs/binary_architectures.npy'

archs = np.load(archs_path)
bin_archs = np.load(bin_archs_path)

#print(archs.shape, bin_archs.shape)
#print(archs[0])
#print()
#print(bin_archs[0])

def test_architecture(arch_code, test_loader):

    arch_code = eval(arch_code)
    net = models.model_entry(cfg, arch_code)
    rank = 0
    world_size = 1
    net = net.cuda()
    print("------", arch_code)
    print()

    net_adv = AttackPGD(net, cfg.attack_param)


def generate_adv_data():
    global cfg, rank, world_size

    cfg = Config.fromfile(args.config)

    np.random.seed(cfg.seed)
    cudnn.benchmark = True
    torch.manual_seed(cfg.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(cfg.seed)

    print('==> Preparing data..')
    testloader = dataset_entry(cfg, args.distributed, args.eval_only)

    #architecture = bin_archs[0]
    for arch in tqdm(bin_archs):
        test_architecture(arch, testloader)
    

