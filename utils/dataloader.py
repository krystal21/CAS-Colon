# -*- coding: utf-8 -*-
"""
时间：2022年03月17日
"""
# encoding: utf-8
import torch
from torch.utils.data import Dataset
from PIL import Image


class DatasetGenerator(Dataset):
    def __init__(self, image_names, labels, transform):
        self.image_names = image_names
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.image_names[index]
        label = self.labels[index]
        image = Image.open(image_name)
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_names)