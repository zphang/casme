import cv2

import torch
import torchvision.transforms as transforms

from casme.model_basics import binarize_mask, get_mask


def get_binarized_mask(input, model, use_p=None):
    mask = get_mask(input, model, use_p=use_p)
    return binarize_mask(mask.clone()), mask.cpu()


def get_masked_images(input, binary_mask, gray_scale=0, return_mask=False):
    with torch.no_grad():
        if gray_scale > 0:
            gray_background = torch.zeros_like(input) + 0.35
            masked_in = binary_mask * input + (1 - binary_mask) * gray_background
            masked_out = (1 - binary_mask) * input + binary_mask * gray_background
        else:
            masked_in = binary_mask * input
            masked_out = (1 - binary_mask) * input

        if return_mask:
            return masked_in, masked_out, binary_mask
        else:
            return masked_in, masked_out


def get_masked_images_v2(batch_x, mask):
    with torch.no_grad():
        masked_in = batch_x * mask
        masked_out = batch_x * (1 - mask)
    return masked_in, masked_out


def inpaint(mask, masked_image):
    l = []
    for i in range(mask.size(0)):
        permuted_image = permute_image(masked_image[i], mul255=True)
        m = mask[i].squeeze().byte().numpy()
        inpainted_numpy = cv2.inpaint(permuted_image, m, 3, cv2.INPAINT_TELEA)  # cv2.INPAINT_NS
        l.append(transforms.ToTensor()(inpainted_numpy).unsqueeze(0))
    inpainted_tensor = torch.cat(l, 0)

    return inpainted_tensor       


def permute_image(image_tensor, mul255 = False):
    with torch.no_grad():
        image = image_tensor.clone().squeeze().permute(1, 2, 0)
        if mul255:
            image *= 255
            image = image.byte()

        return image.numpy()


def per_image_normalization(x, mode):
    if mode is None:
        return x
    assert len(x.shape) == 4
    x_max = x.view(x.shape[0], -1).max(1)[0].view(-1, 1, 1, 1) + 1e-6
    x_min = x.view(x.shape[0], -1).min(1)[0].view(-1, 1, 1, 1)
    if mode == "-1_1":
        normalized_x = (x - x_min) / (x_max - x_min).view(-1, 1, 1, 1) * 2 - 1
    elif mode == "0_1":
        normalized_x = (x - x_min) / (x_max - x_min).view(-1, 1, 1, 1)
    else:
        raise KeyError(mode)
    return normalized_x
