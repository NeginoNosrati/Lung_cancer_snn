
import random
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMG_SIZE = 128 



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transform(train: bool):
    ops = [transforms.Grayscale(num_output_channels=1),
           transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if train:
        ops += [transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2)]
    ops += [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    return transforms.Compose(ops)


def get_loaders(data_dir="data", batch_size=32, num_workers=0):
    root = Path(data_dir)
    train_aug = datasets.ImageFolder(root / "train", get_transform(True))
    train_clean = datasets.ImageFolder(root / "train", get_transform(False))
    val = datasets.ImageFolder(root / "val", get_transform(False))
    test = datasets.ImageFolder(root / "test", get_transform(False))
    assert train_aug.classes == val.classes == test.classes, "class folders differ between splits"
    classes = train_aug.classes  

    counts = np.bincount(train_aug.targets, minlength=len(classes))
    weights = torch.tensor(len(train_aug) / (len(classes) * counts), dtype=torch.float32)

    pin = torch.cuda.is_available()

    def mk(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin)

    return {
        "train": mk(train_aug, True),         
        "train_clean": mk(train_clean, False),  
        "val": mk(val, False),
        "test": mk(test, False),
        "classes": classes,
        "class_weights": weights,
        "counts": counts.tolist(),
    }


if __name__ == "__main__":
    L = get_loaders()
    print("classes:", L["classes"])
    print("train counts:", L["counts"])
    print("class weights:", [round(w, 3) for w in L["class_weights"].tolist()])
    x, y = next(iter(L["train"]))
    print("batch:", tuple(x.shape), "range:", float(x.min()), float(x.max()))