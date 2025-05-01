from torchvision import transforms
import torch
from .dataloader import DatasetGenerator
from models.model import get_model
from PIL import Image


transform_list_train = transforms.Compose(
    [
        # transforms.Resize(256),
        # transforms.RandomCrop(224),
        transforms.Resize(330),
        transforms.CenterCrop(299),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-180, +180)),
        # transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

transform_list_val = transforms.Compose(
    [
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.Resize(330),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def build_dataloader(img_list, labels, batch_size, num_workers, set):
    if set == "sup":
        dataset = DatasetGenerator(img_list, labels, transform_list_train)
    elif set == "val":
        dataset = DatasetGenerator(img_list, labels, transform_list_val)
    else:
        raise ValueError("dataset set error")

    if_shuffle = False if set == "val" else True
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=if_shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
    return dataloader


def build_model_sup(backbone, num_classes):
    model = get_model(backbone, num_classes)
    return model
