import json
import os
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import transforms as T
import utils


class UECFOOD100(Dataset):

    def __init__(self, root, transform=T.ToTensor()):
        self.root = root
        self.transform = transform
        self.images = list(sorted(os.listdir(os.path.join(root, "images")),
                                  key=lambda file: int(re.sub('\\D', '', file))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "annotations")),
                                       key=lambda file: int(re.sub('\\D', '', file))))

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, 'images', self.images[idx])
        annotation_path = os.path.join(self.root, "annotations", self.annotations[idx])
        bboxes = []
        labels = []
        with open(annotation_path) as file:
            annotation = json.load(file)
            for item in annotation:
                x1, y1, x2, y2, label = item
                bboxes.append([x1, y1, x2, y2])
                labels.append(label)
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        image_id = torch.tensor([idx])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {'area': areas, 'boxes': bboxes, 'image_id': image_id, 'labels': labels}
        image = Image.open(image_path).convert("RGB")
        image, target = self.transform(image, target)
        return image, target

    def __len__(self):
        return len(self.images)


def get_train_test_loader(
        root, batch_size=2, augment=True, test_ratio=0.2, shuffle=True,
        num_workers=4, collate_fn=utils.collate_fn, pin_memory=False):
    train_transform = T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
    ]) if augment else T.ToTensor()
    train_dataset = UECFOOD100(root=root, transform=train_transform)
    test_dataset = UECFOOD100(root=root)
    size = len(train_dataset)
    indices = list(range(size))
    split = int(np.floor(test_ratio * size))
    if shuffle:
        np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory
    )
    return train_loader, test_loader
