import os
import errno
import shutil
import os.path as osp
import torch.nn.functional as F
import torch
import numpy as np

def get_splits(dataset, seed, mismatch):
    # one can modify the seed to get different splits
    if dataset == 'cifar10':
        if seed == 1:
            shuffled_list = [8, 2, 7, 4, 3, 5, 9, 6, 0, 1]
            knownclass = shuffled_list[:mismatch]
    elif dataset == 'cifar100':
        if seed == 1:
            shuffled_list = [27, 56, 53, 69, 57, 89, 77, 21, 37, 86, 51, 46, 30, 68, 49, 18, 20, 43, 54, 19, 92, 31, 3, 82, 26, 12, 67, 17, 63, 55, 91, 62, 99, 38, 47, 50, 78, 24, 0, 44, 76, 16, 75, 71, 11, 94, 6, 73, 65, 32, 64, 66, 1, 15, 40, 87, 2, 96, 7, 23, 84, 72, 79, 74, 59, 85, 39, 28, 52, 48, 14, 35, 61, 81, 29, 36, 25, 9, 97, 42, 83, 70, 90, 10, 8, 98, 4, 41, 13, 22, 80, 95, 93, 58, 5, 33, 45, 88, 34, 60]
            knownclass = shuffled_list[:mismatch]
    elif dataset == 'tinyimagenet':
        if seed == 1:
            shuffled_list = [74, 38, 136, 85, 145, 132, 102, 91, 156, 97, 149, 143, 168, 11, 73, 181, 8, 76, 148, 125, 51, 19, 45, 66, 141, 12, 96, 167, 90, 104, 103, 35, 187, 126, 124, 95, 182, 180, 99, 109, 1, 117, 167, 52, 183, 77, 50, 41, 34, 189, 29, 142, 59, 30, 60, 100, 33, 84, 14, 63, 92, 123, 8, 15, 120, 83, 169, 13, 137, 72, 94, 21, 5, 57, 61, 16, 47, 65, 93, 48, 128, 124, 123, 3, 40, 13, 64, 37, 31, 195, 106, 67, 17, 175, 53, 80, 186, 28, 115, 35, 98, 42, 156, 4, 159, 196, 6, 170, 49, 87, 21, 193, 31, 18, 79, 110, 179, 184, 43, 20, 133, 54, 71, 64, 162, 157, 121, 101, 190, 11, 124, 39, 9, 171, 27, 122, 26, 137, 106, 155, 190, 122, 182, 145, 125, 198, 58, 162, 149, 42, 87, 120, 103, 29, 105, 71, 78, 170, 194, 62, 125, 7, 82, 174, 192, 84, 106, 196, 189, 48, 37, 122, 40, 179, 21]
            knownclass = shuffled_list[:mismatch]
    return knownclass


def lab_conv(knownclass, label):
    knownclass = sorted(knownclass)
    label_convert = torch.zeros(len(label), dtype=int)
    for j in range(len(label)):
        for i in range(len(knownclass)):

            if label[j] == knownclass[i]:
                label_convert[j] = int(knownclass.index(knownclass[i]))
                break
            else:
                label_convert[j] = int(len(knownclass))     
    return label_convert


class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

