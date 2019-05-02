import numpy as np
import scipy.ndimage

import torch

from casme import archs
from casme.criterion import default_apply_mask_func, default_infill_func

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_PARAMETERS = {
    "number_of_classes": 3,
    "num_filters": 16,
    "input_channels": 1,
    "first_layer_kernel_size": 7,
    "first_layer_conv_stride": 2,
    "first_pool_size": 3,
    "first_pool_stride": 2,
    "blocks_per_layer_list": [3, 4, 6, 3],
    "block_strides_list": [1, 2, 2, 2],
    "block_fn": "bottleneck",
}


def old_load_model(casm_path):
    name = casm_path.split('/')[-1].replace('.chk', '')

    print("\n=> Loading model localized in '{}'".format(casm_path))
    classifier = archs.resnet50shared()
    checkpoint = torch.load(casm_path)

    classifier.load_state_dict(checkpoint['state_dict_classifier'])
    classifier.eval().to(device)

    masker = archs.masker(
        add_prob_layers=getattr(checkpoint["args"], "add_prob_layers", None)
    )
    print(checkpoint["args"])
    masker.load_state_dict(checkpoint['state_dict_masker'])
    masker.eval().to(device)
    print("=> Model loaded.")

    return {'classifier': classifier, 'masker': masker, 'name': name}


def icasme_load_model(casm_path):
    name = casm_path.split('/')[-1].replace('.chk', '')

    print("\n=> Loading model localized in '{}'".format(casm_path))
    classifier = archs.resnet50shared()
    checkpoint = torch.load(casm_path)

    classifier.load_state_dict(checkpoint['state_dict_classifier'])
    classifier.eval().to(device)

    masker = archs.masker(
        add_prob_layers=getattr(checkpoint["args"], "add_prob_layers", None)
    )
    print(checkpoint["args"])
    masker.load_state_dict(checkpoint['state_dict_masker'])
    masker.eval().to(device)
    print("=> Model loaded.")

    infiller = archs.InfillerCNN(
        4, 3, [32, 64, 128, 256],
    )
    infiller.load_state_dict(checkpoint['state_dict_infiller'])
    infiller.eval().to(device)

    return {'classifier': classifier, 'masker': masker, 'infiller': infiller,
            'name': name}


def get_masks_and_check_predictions(input, target, model):
    with torch.no_grad():
        input, target = torch.tensor(input), torch.tensor(target)
        mask, output = get_mask(input, model, get_output=True)

        rectangular = binarize_mask(mask.clone())

        for id in range(mask.size(0)):
            if rectangular[id].sum() == 0:
                continue
            rectangular[id] = get_rectangular_mask(rectangular[id].squeeze().numpy())

        target = target.to(device)
        _, max_indexes = output.data.max(1)
        isCorrect = target.eq(max_indexes)

        return mask.squeeze().cpu().numpy(), rectangular.squeeze().cpu().numpy(), isCorrect.cpu().numpy() 


def get_mask(input, model, use_p=None, get_output=False):
    with torch.no_grad():
        input = input.to(device)
        classifier_output, layers = model['classifier'](input, return_intermediate=True)
        masker_output = model['masker'](layers, use_p=use_p)
        if get_output:
            return masker_output, classifier_output
        else:
            return masker_output


def get_infilled(x, mask, infiller):
    with torch.no_grad():
        masked_x = default_apply_mask_func(x=x, mask=mask)
        generated = infiller(masked_x.detach(), mask.detach())
        infilled = default_infill_func(masked_x, mask, generated)
    return generated, infilled


def binarize_mask(mask):
    with torch.no_grad():
        avg = mask.view(mask.size(0), -1).mean(dim=1)
        flat_mask = mask.cpu().view(mask.size(0), -1)
        binarized_mask = torch.zeros_like(flat_mask)
        for i in range(mask.size(0)):
            kth = 1 + int((flat_mask[i].size(0) - 1) * (1 - avg[i].item()) + 0.5)
            th, _ = torch.kthvalue(flat_mask[i], kth)
            th.clamp_(1e-6, 1 - 1e-6)
            binarized_mask[i] = flat_mask[i].gt(th).float()
        binarized_mask = binarized_mask.view(mask.size())

        return binarized_mask.to(device)


def get_largest_connected(m):
    mask, num_labels = scipy.ndimage.label(m)
    largest_label = np.argmax(np.bincount(
        mask.reshape(-1), weights=m.reshape(-1)))
    largest_connected = (mask == largest_label)

    return largest_connected


def get_bounding_box(m):
    x = m.any(1)
    y = m.any(0)
    xmin = np.argmax(x)
    xmax = np.argmax(np.cumsum(x))
    ymin = np.argmax(y)
    ymax = np.argmax(np.cumsum(y))
    with torch.no_grad():
        box_mask = torch.zeros(224, 224).to(device)
        box_mask[xmin:xmax+1, ymin:ymax+1] = 1

        return box_mask


def get_rectangular_mask(m):
    return get_bounding_box(get_largest_connected(m))
