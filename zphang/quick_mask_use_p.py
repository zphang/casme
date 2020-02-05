import tqdm
import numpy as np
import pandas as pd

import torch
import torchvision.transforms as transforms

import casme.tasks.imagenet.utils as imagenet_utils
import casme.casme_utils as casme_utils
from casme.utils.torch_utils import ImagePathDataset
from casme.model_basics import casme_load_model, binarize_mask

import zconf


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    casm_path = zconf.attr(default="best")
    config_path = zconf.attr(type=str)
    use_p = zconf.attr(type=float)
    output_path = zconf.attr(default=None)


args = RunConfiguration.run_cli_json_prepend()

data_loader = torch.utils.data.DataLoader(
    ImagePathDataset.from_path(
        config_path=args.config_path,
        transform=transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            imagenet_utils.NORMALIZATION,
        ]),
        return_paths=True,
    ),
    batch_size=72, shuffle=False, num_workers=4, pin_memory=False
)

model = casme_load_model(
    args.casm_path,
    classifier_load_mode="pickled",
    verbose=False,
)


prop_in_ls = []
for i, ((input_, target), paths) in enumerate(tqdm.tqdm(data_loader)):
    input_ = input_.cuda()
    mask_in = casme_utils.get_mask(input_, model, use_p=args.use_p)
    prop_in = binarize_mask(mask_in.clone()).view(input_.shape[0], -1).mean(1).cpu().numpy()
    prop_in_ls.append(prop_in)
full_prop_in = np.concatenate(prop_in_ls)

torch.save(full_prop_in, args.output_path)
