import cv2
import numpy as np

import torch
import torchvision.transforms as transforms

from model_basics import binarize_mask, get_mask


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


def inpaint(mask, masked_image):
    l = []
    for i in range(mask.size(0)):
        permuted_image = permute_image(masked_image[i], mul255=True)
        m = mask[i].squeeze().byte().numpy()
        inpainted_numpy = cv2.inpaint(permuted_image, m, 3, cv2.INPAINT_TELEA)  # cv2.INPAINT_NS
        l.append(transforms.ToTensor()(np.expand_dims(inpainted_numpy, 2)).unsqueeze(0))
    inpainted_tensor = torch.cat(l, 0)

    return inpainted_tensor       


def permute_image(image_tensor, mul255 = False):
    with torch.no_grad():
        # image = image_tensor.clone().squeeze().permute(1, 2, 0)
        image = image_tensor.clone().permute(1, 2, 0)
        if mul255:
            image *= 255
            image = image.byte()

        return image.numpy()
