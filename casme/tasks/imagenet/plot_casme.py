import os
import random

import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from casme.model_basics import casme_load_model
from casme.utils.torch_utils import ImagePathDataset
from casme.casme_utils import get_binarized_mask, get_masked_images, permute_image
import casme.tasks.imagenet.utils as imagenet_utils
import casme.core as core
import casme.archs as archs

import pyutils.io as io
import zconf


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    val_json = zconf.attr(help='val_json path')
    casm_path = zconf.attr(help='model_checkpoint')
    workers = zconf.attr(default=4, type=int, help='number of data loading workers (default: 4)')
    resize = zconf.attr(default=256, type=int, help='resize parameter (default: 256)')
    columns = zconf.attr(default=7, type=int,
                         help='number of consecutive images plotted together,'
                              ' one per column (default: 7, recommended 4 to 7)')
    plots = zconf.attr(default=16, type=int,
                       help='number of different plots generated (default: 16, -1 to generate all of them)')
    seed = zconf.attr(default=931001, type=int, help='random seed that is used to select images')
    plots_path = zconf.attr(default='', help='directory for plots')


def get_infiller(model, device):
    if model["checkpoint"]["args"].get("infiller_model", None):
        return archs.get_infiller(
            model["checkpoint"]["args"]["infiller_model"]
        ).to(device)
    else:
        return None


def write_html(img_path_ls, output_path):
    s = "<html><body>"
    for img_path in img_path_ls:
        s += "<img src='{}' style='max-width: 100%'><hr>".format(img_path)
    s += "</body></html>"
    io.write_file(s, output_path)


def get_infilled(infiller, masked_in_x, masked_out_x, binary_mask, x):
    if infiller is not None:
        infilled_masked_in = core.infill_masked_in(
            infiller=infiller, masked_in_x=masked_in_x,
            mask=binary_mask, x=x,
        )
        infilled_masked_out = core.infill_masked_out(
            infiller=infiller, masked_out_x=masked_out_x,
            mask=binary_mask, x=x,
        )
        return infilled_masked_in, infilled_masked_out
    else:
        return masked_in_x, masked_out_x


def main(args: RunConfiguration):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_config = io.read_json(args.val_json)
    data_config["samples"] = random.choices(
        data_config["samples"],
        k=args.columns * args.plots,
    )

    # data loader without normalization
    data_loader = torch.utils.data.DataLoader(
        ImagePathDataset(
            config=data_config,
            transform=transforms.Compose([
                transforms.Resize(args.resize),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                imagenet_utils.NORMALIZATION,
            ])
        ),
        batch_size=args.columns, shuffle=False, num_workers=args.workers, pin_memory=False,
    )

    model = casme_load_model(args.casm_path)
    infiller = get_infiller(model, device)
    torch.set_grad_enabled(False)

    dir_path = args.plots_path
    os.makedirs(dir_path, exist_ok=True)

    img_path_ls = []
    with PdfPages(os.path.join(dir_path, "plots.pdf")) as pdf:
        for i, (x, _) in enumerate(data_loader):
            images = images.to(device)

            # === get mask and masked images
            binary_mask, soft_mask = get_binarized_mask(x, model)
            soft_masked_image = x * soft_mask
            masked_in_x, masked_out_x = get_masked_images(images, binary_mask, 0.0)
            infilled_masked_in, infilled_masked_out = get_infilled(
                infiller=infiller, masked_in_x=masked_in_x, masked_out_x=masked_out_x,
                binary_mask=binary_mask, x=x,
            )

            # === setup plot
            fig, axes = plt.subplots(6, args.columns, figsize=(10, 10))
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
                axes[1, col].imshow(permute_image(masked_in_x[col]))
                axes[2, col].imshow(permute_image(masked_out_x[col]))
                axes[3, col].imshow(permute_image(infilled_masked_in[col]))
                axes[4, col].imshow(permute_image(infilled_masked_out[col]))
                axes[5, col].imshow(permute_image(soft_masked_image[col]))

            for ax in axes.flatten():
                ax.set_xticks([])
                ax.set_yticks([])

            axes[0, 0].set_ylabel("Original", fontsize=6)
            axes[1, 0].set_ylabel("Masked In", fontsize=6)
            axes[2, 0].set_ylabel("Masked Out", fontsize=6)
            axes[3, 0].set_ylabel("Masked In+Infill", fontsize=6)
            axes[4, 0].set_ylabel("Masked Out+Infill", fontsize=6)
            axes[5, 0].set_ylabel("Soft Masked Out", fontsize=6)

            file_name = "{}.png".format(i)
            path = os.path.join(dir_path, file_name)
            img_path_ls.append(file_name)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            pdf.savefig()
            plt.show()
            plt.clf()
            plt.gcf()
            plt.close('all')
            print('plotted to {}.'.format(path))
    write_html(img_path_ls, os.path.join(dir_path, "index.html"))


if __name__ == '__main__':
    main(args=RunConfiguration.run_cli_json_prepend())
