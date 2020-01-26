import tqdm
import numpy as np
import pandas as pd

import torch
import torchvision.transforms as transforms

import casme.tasks.imagenet.utils as imagenet_utils
import casme.casme_utils as casme_utils
from casme.utils.torch_utils import ImagePathDataset
from casme.model_basics import casme_load_model

import zconf


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    casm_path = zconf.attr(default="best")
    output_path = zconf.attr(default=None)


args = RunConfiguration.run_cli_json_prepend()

data_loader = torch.utils.data.DataLoader(
    ImagePathDataset.from_path(
        config_path="/gpfs/data/geraslab/zphang/working/1912/29_new_metadata/train_val.json",
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


value_counts_ls = []
for i, ((input_, target), paths) in enumerate(tqdm.tqdm(data_loader)):
    input_ = input_.cuda()
    mask_in = casme_utils.get_mask(input_, model, use_p=None)
    EPS = 1e-4
    bucketed = np.floor((mask_in.view(-1).cpu().numpy() - EPS) * 1000).clip(0)
    value_counts_ls.append(
        pd.Series(np.floor((mask_in.view(-1).cpu().numpy() - EPS) * 1000).clip(0).astype(int)).value_counts()
    )

torch.save(value_counts_ls, args.output_path)
