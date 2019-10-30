from torchvision.datasets.folder import has_file_allowed_extension
import numpy as np
import pyutils.io as io
import os

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def find_classes(base_path):
    classes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(fol, class_to_idx, extensions):
    images = []
    fol = os.path.expanduser(fol)
    for target in sorted(os.listdir(fol)):
        d = os.path.join(fol, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def generate_jsons(train_path, val_path, num_per_class_in_a, output_base_path, seed=1234):
    random_state = np.random.RandomState(seed=seed)
    classes, class_to_idx = find_classes(train_path)
    samples = make_dataset(train_path, class_to_idx, IMG_EXTENSIONS)
    random_state.shuffle(samples)

    # Train
    class_dict = {}
    for path, class_idx in samples:
        if class_idx not in class_dict:
            class_dict[class_idx] = []
        class_dict[class_idx].append((path, class_idx))

    samples_a, samples_b = [], []
    for class_idx in range(len(class_dict)):
        class_samples = class_dict[class_idx]
        chosen = set(random_state.choice(np.arange(len(class_samples)),
                                         num_per_class_in_a, replace=False))
        for i, sample in enumerate(class_samples):
            if i in chosen:
                samples_a.append(sample)
            else:
                samples_b.append(sample)

    io.write_json(
        {
            "root": train_path,
            "samples": samples_a,
            "classes": classes,
            "class_to_idx": class_to_idx,
        },
        os.path.join(output_base_path, "train_val.json"),
    )
    io.write_json(
        {
            "root": train_path,
            "samples": samples_b,
            "classes": classes,
            "class_to_idx": class_to_idx,
        },
        os.path.join(output_base_path, "train_train.json"),
    )

    # Val
    classes, class_to_idx = find_classes(val_path)
    val_samples = make_dataset(val_path, class_to_idx, IMG_EXTENSIONS)
    io.write_json(
        {
            "root": val_path,
            "samples": val_samples,
            "classes": classes,
            "class_to_idx": class_to_idx,
        },
        os.path.join(output_base_path, "val.json"),
    )
