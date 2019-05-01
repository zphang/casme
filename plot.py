import argparse
import numpy as np
import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib
import matplotlib.pyplot as plt

from casme.model_basics import old_load_model
from casme.utils import get_binarized_mask, get_masked_images, inpaint, permute_image


matplotlib.use('Agg')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--casm-path', default='',
                    help='path to model that generate masks')

parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resize', default=256, type=int,
                    help='resize parameter (default: 256)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')

parser.add_argument('--columns', default=7, type=int,
                    help='number of consecutive images plotted together,'
                         ' one per column (default: 7, recommended 4 to 7)')
parser.add_argument('--plots', default=16, type=int,
                    help='number of different plots generated (default: 16, -1 to generate all of them)')
parser.add_argument('--seed', default=931001, type=int,
                    help='random seed that is used to select images')
parser.add_argument('--plots-path', default='',
                    help='directory for plots')

args = parser.parse_args()

if args.columns > args.batch_size:
    args.columns = args.batch_size


def main():
    global args

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=mean,
                                     std=std)
    denormalize = transforms.Normalize(mean=-mean / std,
                                       std=1 / std)
    # data loader without normalization
    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(args.data, 'val'), transforms.Compose([
            transforms.Resize(args.resize),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    model = old_load_model(args.casm_path)

    perm = np.random.RandomState(seed=args.seed).permutation(len(data_loader))
    if args.plots > 0:
        perm = perm[:args.plots]
        print('List of sampled batches:', sorted(perm))

    dir_path = args.plots_path
    os.makedirs(dir_path, exist_ok=True)

    for i, (input_, target) in enumerate(data_loader):
        print('{} '.format(i), end='', flush=True)
        if i not in perm:
            print('skipped.')
            continue

        # === normalize first few images
        normalized_input = input_.clone()
        for k in range(args.columns):
            normalize(normalized_input[k])

        # === get mask and masked images
        binary_mask, soft_mask = get_binarized_mask(normalized_input, model)
        soft_masked_image = normalized_input * soft_mask
        for j in range(soft_masked_image.size(0)):
            denormalize(soft_masked_image[j])
        masked_in, masked_out = get_masked_images(input_, binary_mask, 0.35)
        inpainted = inpaint(binary_mask, masked_out)

        # === setup plot
        fig, axes = plt.subplots(5, args.columns)
        if args.columns == 4:
            fig.subplots_adjust(bottom=-0.02, top=1.02, wspace=0.05, hspace=0.05)
        if args.columns == 5:
            fig.subplots_adjust(top=0.92, wspace=0.05, hspace=0.05)
        if args.columns == 6:
            fig.subplots_adjust(top=0.8, wspace=0.05, hspace=0.05)
        if args.columns == 7:
            fig.subplots_adjust(top=0.7, wspace=0.05, hspace=0.05)

        # === plot
        for col in range(args.columns):
            axes[0, col].imshow(permute_image(input_[col]))
            axes[1, col].imshow(permute_image(masked_in[col]))
            axes[2, col].imshow(permute_image(masked_out[col]))
            axes[3, col].imshow(permute_image(inpainted[col]))
            axes[4, col].imshow(permute_image(soft_masked_image[col]))

        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        path = os.path.join(dir_path, str(i) + '.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.gcf()
        plt.close('all')
        print('plotted to {}.'.format(path))


if __name__ == '__main__':
    main()
