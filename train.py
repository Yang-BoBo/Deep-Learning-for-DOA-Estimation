import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch

from config import Config
from data import Cifar10Dataset

from model import Network


def train():
  # Configuration settings
  cfg = Config()

  # Load dataset
  dataset = Cifar10Dataset(cfg.gt_dir, split='train')
  dataloader = DataLoader(dataset, batch_size=cfg.batch_size_train, shuffle=True, num_workers=4)

  # Initialize network
  model = Network(cfg)
  model.train()
  if cfg.enable_cuda:
    model = model.cuda()

  # Initialize optimizer
  optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.lr_momentum, weight_decay=cfg.weight_decay)

  # Loop over images
  running_loss = 0.0
  i = 0
  print("Starting training...")
  while i < cfg.num_iterations:
    for (imgs, labels) in dataloader:
      if i > cfg.num_iterations:
        break
      if cfg.enable_cuda:
        imgs = imgs.cuda()
        labels = labels.cuda()

      optimizer.zero_grad()
      out = model(imgs, labels)
      loss = out['loss']

      running_loss += out['loss'].item()

      # Apply back-propagation
      loss.backward()
      # Take one step with the optimizer
      optimizer.step()

      if i % cfg.log_iterations == 0:
        if i == 0:
          loss_avg = running_loss
        else:
          loss_avg = running_loss/cfg.log_iterations
        print("Iteration {} - Loss: {}".format(i, round(loss_avg, 5)))
        running_loss = 0.0
      i += 1

  print("Finished training.")
  save_path = 'model.pth'
  torch.save(model.state_dict(), save_path)
  print("Saved trained model as {}.".format(save_path))


if __name__ == "__main__":
  train()
