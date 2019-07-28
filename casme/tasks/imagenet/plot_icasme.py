import json
import numpy as np
import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages

from casme.model_basics import (
    icasme_load_model, get_infilled, get_masks_and_check_predictions,
    get_pred_bounding_box,
)
from casme.utils import get_binarized_mask, get_masked_images, inpaint, permute_image
from casme.tasks.imagenet.score_bboxes import compute_agg_loc_scores

import zconf


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        return super(ImageFolderWithPaths, self).__getitem__(index), self.imgs[index][0]


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=mean,
                                     std=std)
    denormalize = transforms.Normalize(mean=-mean / std,
                                       std=1 / std)
    # data loader without normalization
    data_loader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(os.path.join(args.data, 'val'), transforms.Compose([
            transforms.Resize(args.resize),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    with open(args.bboxes_path, "r") as f:
        bboxes = json.loads(f.read())

    model = icasme_load_model(args.casm_path)

    perm = np.random.RandomState(seed=args.seed).permutation(len(data_loader))
    if args.plots > 0:
        perm = perm[:args.plots]
        print('List of sampled batches:', sorted(perm))

    dir_path = args.plots_path
    os.makedirs(dir_path, exist_ok=True)

    with PdfPages(os.path.join(dir_path, "all.pdf")) as pdf:
        for i, ((input_, target), paths) in enumerate(data_loader):
            input_ = input_.to(device)
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
            generated, infilled = get_infilled(normalized_input, soft_mask, model["infiller"])
            continuous, rectangular, is_correct = get_masks_and_check_predictions(input_, target, model)

            for j in range(soft_masked_image.size(0)):
                denormalize(soft_masked_image[j])
                denormalize(generated[j])
                denormalize(infilled[j])
            masked_in, masked_out = get_masked_images(input_, binary_mask, 0.35)
            inpainted = inpaint(binary_mask, masked_out)

            # === setup plot
            fig, axes = plt.subplots(8, args.columns, figsize=(14, 16))
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
                axes[5, col].imshow(permute_image(generated[col]))
                axes[6, col].imshow(permute_image(infilled[col]))
                axes[7, col].imshow(permute_image(input_[col]))

                gt_boxes = bboxes[os.path.basename(paths[col]).split('.')[0]]
                for ax_i in range(7):
                    ax = axes[ax_i, col]
                    for box in gt_boxes:
                        rect = patches.Rectangle(
                            (box["xmin"], box["ymin"]),
                            box["xmax"] - box["xmin"],
                            box["ymax"] - box["ymin"],
                            linewidth=1, edgecolor='r', facecolor='none',
                        )
                        ax.add_patch(rect)

                # Localization
                for box in gt_boxes:
                    rect = patches.Rectangle(
                        (box["xmin"], box["ymin"]),
                        box["xmax"] - box["xmin"],
                        box["ymax"] - box["ymin"],
                        linewidth=1, edgecolor='r', facecolor=(1, 0, 0, 0.3),
                    )
                    axes[7, col].add_patch(rect)

                idx = col
                loc_scores_dict = compute_agg_loc_scores(
                    gt_boxes=gt_boxes,
                    continuous_mask=continuous[idx],
                    rectangular_mask=rectangular[idx],
                    is_correct=is_correct[idx]
                )
                axes[0, col].set_title("{:.3f} / {:.3f} / {:.3f}".format(
                    loc_scores_dict["f1"], loc_scores_dict["le"], loc_scores_dict["om"],
                ), fontsize=8)
                pred_bbox = get_pred_bounding_box(rectangular[col])
                rect = patches.Rectangle(
                    (pred_bbox["xmin"], pred_bbox["ymin"]),
                    pred_bbox["xmax"] - pred_bbox["xmin"],
                    pred_bbox["ymax"] - pred_bbox["ymin"],
                    linewidth=1, edgecolor='g', facecolor=(0, 1, 0, 0.3),
                )
                axes[7, col].add_patch(rect)

            for ax in axes.flatten():
                ax.set_xticks([])
                ax.set_yticks([])

            path = os.path.join(dir_path, str(i) + '.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            pdf.savefig(fig)
            plt.clf()
            plt.gcf()
            plt.close()
            print('plotted to {}.'.format(path))


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    data = zconf.attr(metavar='DIR', help='path to dataset')
    bboxes_path = zconf.attr(help='path to bboxes_json')
    casm_path = zconf.attr(default='', help='path to model that generate masks')

    workers = zconf.attr(default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    resize = zconf.attr(default=256, type=int,
                        help='resize parameter (default: 256)')
    batch_size = zconf.attr(default=128, type=int, help='mini-batch size (default: 128)')

    columns = zconf.attr(default=7, type=int, help='number of consecutive images plotted together,'
                                                   ' one per column (default: 7, recommended 4 to 7)')
    plots = zconf.attr(default=16, type=int,
                       help='number of different plots generated (default: 16, -1 to generate all of them)')
    seed = zconf.attr(default=931001, type=int,
                      help='random seed that is used to select images')
    plots_path = zconf.attr(default='', help='directory for plots')

    def _post_init(self):
        if self.columns > self.batch_size:
            self.columns = self.batch_size


if __name__ == '__main__':
    main(RunConfiguration.run_cli())
