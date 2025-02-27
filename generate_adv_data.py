import pickle

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

archs = np.load(archs_path).tolist()
bin_archs = np.load(bin_archs_path).tolist()

criterion = nn.CrossEntropyLoss()


# print(archs.shape, bin_archs.shape)
# print(archs[0])
# print()
# print(bin_archs[0])

def test(net, net_adv, testloader, adv=False):
    losses = utils.AverageMeter(0)
    top1 = utils.AverageMeter(0)
    top5 = utils.AverageMeter(0)

    logger = logging.getLogger('global_logger')

    net.eval()

    all_fsp_sum = list()

    data_num = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            print(f"batch index: #{batch_idx + 1}")

            data_num += inputs.shape[0]
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs, fsps = net(inputs)
            adv_outputs, adv_fsps, inputs_adv = net_adv(inputs, targets)

            fsps_losses = list()
            for fsp, adv_fsp in zip(fsps, adv_fsps):
                fsps_losses.append((fsp - adv_fsp).norm(dim=(1, 2)).sum().item())

            if len(all_fsp_sum) == 0:
                all_fsp_sum = fsps_losses
            else:
                all_fsp_sum = [all_fsp_sum[i] + fsps_losses[i] for i in range(len(fsps_losses))]

        all_fsp_losses = [all_fsp_sum[i] / data_num for i in range(len(all_fsp_sum))]
        print("**** final:", len(all_fsp_losses), data_num)
        return all_fsp_losses


def test_architecture(arch_code, test_loader):
    # arch_code = eval(arch_code)
    net = models.model_entry(cfg, arch_code)
    net = net.cuda()
    # print("------", arch_code)
    # print()

    net_adv = AttackPGD(net, cfg.attack_param)

    print('==> Testing on Clean Data..')
    return test(net, net_adv, test_loader)
    # print('==> Testing on Adversarial Data..')
    # test(net_adv, test_loader, adv=True)


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

    # architecture = bin_archs[0]
    arch_tuples = []
    with open('fsp_losses.pickle', 'wb') as f:
        pickle.dump(arch_tuples, f)
        print("pickle file created!")

    for i, arch in enumerate(tqdm(bin_archs)):
        print(f"binary arch: #{i + 1}")

        arch_fsps = test_architecture(arch, testloader)

        with open('fsp_losses.pickle', 'rb') as f:
            arch_tuples = pickle.load(f)

        arch_tuples += [(arch, arch_fsps)]

        with open('fsp_losses.pickle', 'wb') as f:
            pickle.dump(arch_tuples, f)
            print("changes saved in pickle file!")


generate_adv_data()
