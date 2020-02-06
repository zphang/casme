import re
import os
import glob
import pandas as pd
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

import pyutils.io as io
import pyutils.datastructures as datastructures

import torch.nn as nn
import torchvision.transforms as transforms

import zphang.utils as utils
import casme.tasks.imagenet.plot_casme as plot_casme
import casme.tasks.imagenet.utils as imagenet_utils
from casme.utils.torch_utils import ImagePathDataset
from casme.model_basics import casme_load_model
from casme.casme_utils import get_binarized_mask, get_masked_images

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


MY_CMAP = LinearSegmentedColormap(
    "MY_CMAP",
    segmentdata={
        "red": [(0, 0.0, 0.0), (1, 1, 1)],
        "green": [(0, 0.0, 0.0), (1, 0.0, 0.0)],
        "blue": [(0, 0.0, 0.0), (1, 0.0, 0.0)],
    },
    N=256,
)


def phase_determiner(s):
    if s.startswith("l"):
        if s.count("_") == 1:
            return "single"
        elif s.count("_") == 4:
            return "double"
    elif s.startswith("use_layer"):
        return "use_layer"
    elif s.startswith("infiller"):
        return "infiller"
    elif s.startswith("gumbel"):
        return "gumbel"
    raise KeyError(s)


def get_best(df, columns, set_index=True):
    new_df = df.loc[df.groupby(columns)["LE"].idxmin()]
    if set_index:
        new_df = new_df.set_index(columns)
    return new_df


def format_to_table(df):
    s = ""
    for i, row in df.iterrows():
        s += f"  & {row['OM'] * 100:.1f} & {row['LE'] * 100:.1f} & {row['F1'] * 100:.1f} "
        s += f"& {row['SM'] * 100:.1f} & {row['avg_mask'] * 100:.1f} \n"
    return s


def get_df(starpath_ls, regex_str):
    path_ls = sorted([x for starpath in starpath_ls for x in glob.glob(starpath)])
    df = utils.load_score(path_ls, regex_str, filename="new_score.json")
    df["phase"] = df["other"].apply(phase_determiner)
    for col in ["avg_mask", "std_mask", "entropy", "tv"]:
        df[col] = df[col].astype(float)
    return df


def df_select(df, select_dict):
    selector = None
    for k, v in select_dict.items():
        this_selector = df[k] == v
        if selector is None:
            selector = this_selector
        else:
            selector &= this_selector
    selected = df[selector]
    return selected


def df_select_one(df, select_dict):
    selected = df_select(df, select_dict)
    assert len(selected) == 1
    return selected.iloc[0]


TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    imagenet_utils.NORMALIZATION,
])


@dataclass
class Input:
    x: torch.Tensor
    y: torch.Tensor
    use_p: torch.Tensor
    paths: list


@dataclass
class ModelWrapper:
    model: nn.Module
    infiller: nn.Module

    @classmethod
    def from_path(cls, path, device):
        model = casme_load_model(path, verbose=False)
        infiller = plot_casme.get_infiller(model, device)
        return cls(model, infiller)


def get_inputs(data_config, i, add_prob_layers=False, use_p_mode=False, use_p_kwargs=None):
    a = add_prob_layers

    class Temp:
        add_prob_layers = a

    modified_data_config = data_config.copy()
    modified_data_config["samples"] = data_config["samples"][i: i + 1]
    data_loader = torch.utils.data.DataLoader(
        ImagePathDataset(
            config=modified_data_config,
            transform=TRANSFORMS,
            return_paths=True,
        ),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=False,
    )
    (x, y), paths = next(iter(data_loader))
    x, paths, use_p = plot_casme.prep_inputs(
        x=x, paths=paths, columns=1,
        device=DEVICE, masker=Temp,
        use_p_mode=use_p_mode,
        use_p_kwargs=use_p_kwargs,
    )
    return Input(x, y, use_p, paths)


@dataclass
class Outputs:
    x: Any
    masked_in_x: Any
    masked_out_x: Any
    infilled_masked_in: Any
    infilled_masked_out: Any
    soft_mask_arr: Any
    binary_mask_arr: Any


def run_model(model_wrapper, inputs, plottable=True):
    x = inputs.x
    binary_mask, soft_mask = get_binarized_mask(
        inputs.x, model_wrapper.model, use_p=inputs.use_p)
    masked_in_x, masked_out_x = get_masked_images(x, binary_mask, 0.0)
    infilled_masked_in, infilled_masked_out = plot_casme.get_infilled(
        infiller=model_wrapper.infiller, masked_in_x=masked_in_x, masked_out_x=masked_out_x,
        binary_mask=binary_mask, x=x,
    )
    # soft_masked_image = x * soft_mask
    soft_mask_arr = soft_mask.squeeze().cpu().numpy()
    binary_mask_arr = binary_mask.squeeze().cpu().numpy()
    x = x[0]
    masked_in_x = masked_in_x[0]
    masked_out_x = masked_out_x[0]
    infilled_masked_in = infilled_masked_in[0]
    infilled_masked_out = infilled_masked_out[0]
    if plottable:
        x = plot_casme.to_plottable(x)
        masked_in_x = plot_casme.to_plottable(masked_in_x)
        masked_out_x = plot_casme.to_plottable(masked_out_x)
        infilled_masked_in = plot_casme.to_plottable(infilled_masked_in)
        infilled_masked_out = plot_casme.to_plottable(infilled_masked_out)
    return Outputs(
        x,
        masked_in_x, masked_out_x,
        infilled_masked_in, infilled_masked_out,
        soft_mask_arr, binary_mask_arr,
    )


def red_colorize(mask):
    return np.stack([mask, np.zeros([224, 224]), np.zeros([224, 224])], axis=2)


def prime_colorize(mask):
    return np.stack([np.zeros([224, 224]), mask, np.zeros([224, 224])], axis=2)


def mask_concatenated(x, mask):
    colored_mask = prime_colorize(mask)
    return np.concatenate([(x*0.8 + colored_mask).clip(0, 1), colored_mask], axis=1)


def multi_sorted_glob(star_path_ls):
    result = []
    for star_path in star_path_ls:
        result += glob.glob(star_path)
    return sorted(result)
