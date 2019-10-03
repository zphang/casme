from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS

import pyutils.io as io


class ImageJsonDataset(VisionDataset):
    def __init__(self, config_path, transform=None, target_transform=None,
                 loader=default_loader):
        config = io.read_json(config_path)
        super().__init__(root=config["root"], transform=transform, target_transform=target_transform)

        self.loader = loader
        self.extensions = IMG_EXTENSIONS

        self.classes = config["classes"]
        self.class_to_idx = config["class_to_idx"]
        self.samples = config["samples"]
        self.targets = [s[1] for s in self.samples]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)
