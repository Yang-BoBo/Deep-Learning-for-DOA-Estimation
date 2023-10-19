import os
import pickle
import numpy as np
import torch
import time
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image


class Cifar10Dataset(Dataset):
  def __init__(self, gt_dir, split='train'):
    assert split in ['train', 'test'], 'Only train and test splits are implemented.'
    assert os.path.exists(gt_dir), 'gt_dir path does not exist: {}'.format(gt_dir)

    self.gt_dir = gt_dir
    self.split = split

    self.gt_dir_img = os.path.join(gt_dir, split)
    self.gt_path_json = os.path.join(gt_dir, "{}_labels.json".format(split))

    with open(self.gt_path_json) as fp:
      self.labels = json.load(fp)
    self.num_images = len(self.labels)

    self.mean = np.array([0.5, 0.5, 0.5]).reshape([3, 1, 1])
    self.std = np.array([0.5, 0.5, 0.5]).reshape([3, 1, 1])

  # Returns the length of the dataset
  def __len__(self):
    return self.num_images

  # Returns a dataset sample given an idx [0, len(dataset))
  def __getitem__(self, idx):
    idx_str = str(idx).zfill(6)

    image_path = os.path.join(self.gt_dir_img, idx_str + '.png')
    image = read_image(image_path)
    image = image / 255.

    label = self.labels[idx_str]
    label = torch.as_tensor(label, dtype=torch.long)

    # Normalize the image
    image = image - self.mean
    image = image / self.std

    image = torch.as_tensor(image, dtype=torch.float)

    return image, label
