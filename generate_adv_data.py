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

#print(archs.shape, bin_archs.shape)
#print(archs[0])
#print()
#print(bin_archs[0])

def test(net, testloader, adv=False):
    losses = utils.AverageMeter(0)
    top1 = utils.AverageMeter(0)
    top5 = utils.AverageMeter(0)

    logger = logging.getLogger('global_logger')

    net.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()

            # here I think we should calculate the FSP of different cells
            if not adv:
                outputs = net(inputs)
            else:
                outputs, inputs_adv = net(inputs, targets)

            #loss = criterion(outputs, targets)
            #prec1, prec5 = utils.accuracy(outputs.data, targets, topk=(1, 5))

            #num = inputs.size(0)
            #losses.update(loss.clone().item(), num)
            #top1.update(prec1.clone().item(), num)
            #top5.update(prec5.clone().item(), num)
            
            #if batch_idx % cfg.report_freq == 0 and rank == 0:
            #    logger.info(
            #        'Test: [{0}/{1}]\t'
		    #'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
		    #'Prec@1: {top1.val:.3f} ({top1.avg:.3f})\t'
		    #'Prec@5: {top5.val:.3f} ({top5.avg:.3f})\t'
            #        .format(batch_idx, len(testloader), loss=losses, top1=top1, top5=top5))

    #final_loss_sum = torch.Tensor([losses.sum]).cuda()
    #final_top1_sum = torch.Tensor([top1.sum]).cuda()
    #final_top5_sum = torch.Tensor([top5.sum]).cuda()
    #total_num = torch.Tensor([losses.count]).cuda()
    #if args.distributed:
    #    torch.distributed.all_reduce(final_loss_sum)
    #    torch.distributed.all_reduce(final_top1_sum)
    #    torch.distributed.all_reduce(final_top5_sum)
    #    torch.distributed.all_reduce(total_num)
    #final_loss = final_loss_sum.item() / total_num.item()
    #final_top1 = final_top1_sum.item() / total_num.item()
    #final_top5 = final_top5_sum.item() / total_num.item()

    #logger.info(' * Prec@1 {:.3f}\tPrec@5 {:.3f}\tLoss {:.3f}\t'.format(final_top1, final_top5, final_loss))

    #return final_top1

def test_architecture(arch_code, test_loader):

    #arch_code = eval(arch_code)
    net = models.model_entry(cfg, arch_code)
    net = net.cuda()
    #print("------", arch_code)
    #print()

    net_adv = AttackPGD(net, cfg.attack_param)

    print('==> Testing on Clean Data..')
    test(net, test_loader)
    print('==> Testing on Adversarial Data..')
    test(net_adv, test_loader, adv=True)


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
    

generate_adv_data()