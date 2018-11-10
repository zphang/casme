import argparse
import numpy as np
import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from casme.model_basics import load_model, device
from casme.utils import get_binarized_mask, get_masked_images, inpaint, permute_image


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--casm-path', default='',
                    help='path to model that generate masks')

parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resize', default=256, type=int,
                    help='resize parameter (default: 256)')
parser.add_argument('--plots', default=16, type=int,
                    help='number of different plots generated (default: 16, -1 to generate all of them)')
parser.add_argument('--seed', default=931001, type=int,
                    help='random seed that is used to select images')
parser.add_argument('--plots-path', default='',
                    help='directory for plots')
parser.add_argument('--percentages', default=range(25, 80, 5), nargs='+', type=int,
                    help='percentage of image to target for p')

args = parser.parse_args()


def main():
    global args

    num_p = len(args.percentages)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # data loader without normalization
    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(args.data, 'val'), transforms.Compose([
            transforms.Resize(args.resize),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])),
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=False)

    model = load_model(args.casm_path)

    perm = np.random.RandomState(seed=args.seed).permutation(len(data_loader))
    if args.plots > 0:
        perm = perm[:args.plots]
        print('List of sampled batches:', sorted(perm))

    dir_path = args.plots_path
    os.makedirs(dir_path, exist_ok=True)

    for i, (input, target) in enumerate(data_loader):
        print('{} '.format(i), end='', flush=True)
        if i not in perm:
            print('skipped.')
            continue

        assert input.shape[0] == 1
        one_img = input[0]
        copies_of_img = one_img.expand(num_p, 3, 224, 224)

        p = (torch.Tensor(args.percentages) / 100).to(device)

        # normalize first few images
        normalized_input = copies_of_img.clone()
        for id in range(num_p):
            normalize(normalized_input[id])

        # get mask and masked images
        binary_mask, soft_mask = get_binarized_mask(normalized_input, model, p=p)
        masked_in, masked_out, binary_mask = \
            get_masked_images(copies_of_img, binary_mask, 0.35, return_mask=True)
        inpainted = inpaint(binary_mask, masked_out)
        fraction_masked = binary_mask.squeeze(1).mean(1).mean(1).numpy()

        fig, axes = plt.subplots(4, num_p)

        for col in range(num_p):
            axes[0, col].imshow(permute_image(copies_of_img[col]))
            axes[1, col].imshow(permute_image(masked_in[col]))
            axes[2, col].imshow(permute_image(masked_out[col]))
            axes[3, col].imshow(permute_image(inpainted[col]))

            axes[0, col].set_title("{:.3f}".format(args.percentages[col] / 100), fontsize=5)
            axes[1, col].set_title("{:.3f}".format(fraction_masked[col]), fontsize=5)

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
