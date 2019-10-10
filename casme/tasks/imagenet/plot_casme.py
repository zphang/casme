import numpy as np
import os

import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from casme.model_basics import casme_load_model
from casme.utils.torch_utils import ImageJsonDataset
from casme.casme_utils import get_binarized_mask, get_masked_images, inpaint, permute_image
import casme.tasks.imagenet.utils as imagenet_utils

import zconf


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    val_json = zconf.attr(help='val_json path')
    casm_path = zconf.attr(help='model_checkpoint')
    workers = zconf.attr(default=4, type=int, help='number of data loading workers (default: 4)')
    resize = zconf.attr(default=256, type=int, help='resize parameter (default: 256)')
    batch_size = zconf.attr(default=128, type=int, help='mini-batch size (default: 256)')
    columns = zconf.attr(default=7, type=int,
                         help='number of consecutive images plotted together,'
                              ' one per column (default: 7, recommended 4 to 7)')
    plots = zconf.attr(default=16, type=int,
                       help='number of different plots generated (default: 16, -1 to generate all of them)')
    seed = zconf.attr(default=931001, type=int, help='random seed that is used to select images')
    plots_path = zconf.attr(default='', help='directory for plots')

    def _post_init(self):
        if self.columns > self.batch_size:
            self.columns = self.batch_size


def main(args: RunConfiguration):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # data loader without normalization
    data_loader = torch.utils.data.DataLoader(
        ImageJsonDataset(
            args.val_json,
            transforms.Compose([
                transforms.Resize(args.resize),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        ),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False,
    )

    model = casme_load_model(args.casm_path)

    perm = np.random.RandomState(seed=args.seed).permutation(len(data_loader))
    if args.plots > 0:
        perm = perm[:args.plots]
        print('List of sampled batches:', sorted(perm))

    dir_path = args.plots_path
    os.makedirs(dir_path, exist_ok=True)

    for i, (images, target) in enumerate(data_loader):
        images = images.to(device)
        print('{} '.format(i), end='', flush=True)
        if i not in perm:
            print('skipped.')
            continue

        # === normalize first few images
        normalized_input = images.clone()
        for k in range(args.columns):
            imagenet_utils.NORMALIZATION(normalized_input[k])

        # === get mask and masked images
        binary_mask, soft_mask = get_binarized_mask(normalized_input, model)
        soft_masked_image = normalized_input * soft_mask

        for j in range(soft_masked_image.size(0)):
            imagenet_utils.DENORMALIZATION(soft_masked_image[j])
        masked_in, masked_out = get_masked_images(images, binary_mask, 0.35)
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
            axes[0, col].imshow(permute_image(images[col]))
            axes[1, col].imshow(permute_image(masked_in[col]))
            axes[2, col].imshow(permute_image(masked_out[col]))
            axes[3, col].imshow(permute_image(inpainted[col]))
            axes[4, col].imshow(permute_image(soft_masked_image[col]))

        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        axes[0, 0].set_ylabel("Original")
        axes[1, 0].set_ylabel("Masked In")
        axes[2, 0].set_ylabel("Masked Out")
        axes[3, 0].set_ylabel("Bad Inpaint")
        axes[4, 0].set_ylabel("Soft Masked Out")

        path = os.path.join(dir_path, str(i) + '.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.gcf()
        plt.close('all')
        print('plotted to {}.'.format(path))


if __name__ == '__main__':
    main(args=RunConfiguration.run_cli_json_prepend())
