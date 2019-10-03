import torch

import torchvision.transforms as transforms
from casme.utils.torch_utils import ImageJsonDataset


NORMALIZATION = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


def get_data_loaders(train_json, val_json, batch_size, workers):
    if train_json:
        train_loader = torch.utils.data.DataLoader(
            ImageJsonDataset(
                config_path=train_json,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    NORMALIZATION,
                ])),
            batch_size=batch_size, shuffle=True, num_workers=workers,
            pin_memory=False, sampler=None,
        )
    else:
        train_loader = None

    if val_json:
        val_loader = torch.utils.data.DataLoader(
            ImageJsonDataset(
                config_path=val_json,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    NORMALIZATION,
                ])),
            batch_size=batch_size, shuffle=False, num_workers=workers,
            pin_memory=False,
        )

    else:
        val_loader = None

    return train_loader, val_loader
