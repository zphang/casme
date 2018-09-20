import argparse
import numpy as np
import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import patch_classification.patch_model.torch.patch_data_torch as patch_data
from torch.utils.data import DataLoader
from model_basics import load_model
from utils import get_binarized_mask, get_masked_images, inpaint, permute_image
import utilities.logger as logging_module

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--casm-path', default='',
                    help='path to model that generate masks')

parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resize', default=256, type=int,
                    help='resize parameter (default: 256)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')

parser.add_argument('--columns', default=7, type=int,
                    help='number of consecutive images plotted together, one per column (default: 7, recommended 4 to 7)')
parser.add_argument('--plots', default=16, type=int, 
                    help='number of different plots generated (default: 16, -1 to generate all of them)')
parser.add_argument('--seed', default=931001, type=int, 
                    help='random seed that is used to select images')
parser.add_argument('--plots-path', default='',
                    help='directory for plots')

args = parser.parse_args()

if args.columns > args.batch_size:
    args.columns = args.batch_size


DATA_CONFIG = {
    "num_parts": {
        "training": {'malignant': 30, 'benign': 30, 'unknown': 100, 'outside': 100},
        "validation": {'malignant': 1, 'benign': 1, 'unknown': 1, 'outside': 1},
        "test": {'malignant': 1, 'benign': 1, 'unknown': 1, 'outside': 1},
    },
    "samples_per_epoch": {
        "training": {'malignant': 100, 'benign': 200, 'unknown': 51700, 'outside': 48000},
        # "training": {'malignant': 100 , 'benign': 200, 'unknown': 100, 'outside': 100},
        "validation": {'malignant': 10, 'benign': 10, 'unknown': 10, 'outside': 10},
        # "validation": {'malignant': -1, 'benign': -1, 'unknown': -1, 'outside': -1},
        "test": {'malignant': -1, 'benign': -1, 'unknown': -1, 'outside': -1},
    },
    "parts_to_load": {
        "training": {'malignant': 1, 'benign': 1, 'unknown': 6, 'outside': 5},
        # "training": {'malignant': 1, 'benign': 1, 'unknown': 1, 'outside': 1},
        "validation": {'malignant': 1, 'benign': 1, 'unknown': 1, 'outside': 1},
        "test": {'malignant': 1, 'benign': 1, 'unknown': 1, 'outside': 1},
    },
    "paths": {
        "training": "/gpfs/data/geraslab/Nan/data/segmentation_patches/patches_by_class_0907/training/",
        "validation": "/gpfs/data/geraslab/Nan/data/segmentation_patches/patches_by_class_0907/validation/",
        "test": "/gpfs/data/geraslab/Nan/data/segmentation_patches/patches_by_class_0907/test/",
    },
    "label_list": ["malignant", "benign", "unknown", "outside"],
}


def main():
    global args

    logger = logging_module.PrintLogger()
    dataset = patch_data.PickleDataset("validation", data_config=DATA_CONFIG, logger=logger)
    data_loader = DataLoader(dataset, args.batch_size)
    dataset.new_epoch()
    
    model = load_model(args.casm_path)
    
    perm = np.random.RandomState(seed=args.seed).permutation(len(data_loader))
    if args.plots > 0:
        perm = perm[:args.plots]
        print('List of sampled batches:', sorted(perm))

    dir_path = args.plots_path
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for i, (input, target) in enumerate(data_loader):
        print('{} '.format(i), end='', flush=True)
        if i not in perm:
            print('skipped.')
            continue

        ## normalize first few images
        normalized_input = input.clone()

        ## get mask and masked images
        binary_mask, soft_mask = get_binarized_mask(normalized_input, model)
        soft_masked_image = normalized_input * soft_mask
        masked_in, masked_out = get_masked_images(input, binary_mask, 0.35)
        inpainted = inpaint(binary_mask, masked_out)

        ## setup plot
        fig, axes = plt.subplots(5, args.columns)
        if args.columns == 4:
            fig.subplots_adjust(bottom=-0.02, top=1.02, wspace=0.05, hspace=0.05)
        if args.columns == 5:
            fig.subplots_adjust(top=0.92, wspace=0.05, hspace=0.05)
        if args.columns == 6:
            fig.subplots_adjust(top=0.8, wspace=0.05, hspace=0.05)
        if args.columns == 7:
            fig.subplots_adjust(top=0.7, wspace=0.05, hspace=0.05)

        ## plot
        for col in range(args.columns):
            axes[0, col].set_title(target[col])
            axes[0, col].imshow(permute_image(input[col]), cmap="gray")
            axes[1, col].imshow(permute_image(masked_in[col]), cmap="gray")
            axes[2, col].imshow(permute_image(masked_out[col]), cmap="gray")
            axes[3, col].imshow(permute_image(inpainted[col]), cmap="gray")
            axes[4, col].imshow(permute_image(soft_masked_image[col]), cmap="gray")

        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        path = os.path.join(dir_path,str(i) + '.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.gcf()
        plt.close('all')
        print('plotted to {}.'.format(path))


if __name__ == '__main__':
    main()
