import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    # 7x7 conv with 3 input channels and 12 output channels
    self.conv1 = nn.Conv2d(3, 12, 7)
    # 2x2 max pooling
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(12, 24, 3)
    # Fully-connected/Linear layer with 24*5*5 inputs and 128 outputs
    self.fc1 = nn.Linear(24 * 5 * 5, 128)

  def forward(self, x):
    # Apply ReLU activation after convolution
    x = F.relu(self.conv1(x))
    x = self.pool(x)
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    # Flatten from [Nb, 24, 5, 5] to [Nb, 600]
    x = torch.reshape(x, [-1, 24 * 5 * 5])
    x = F.relu(self.fc1(x))
    return x


class Classifier(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.num_classes = cfg.num_classes
    self.classifier = nn.Linear(128, self.num_classes)

  def forward(self, x):
    x = self.classifier(x)
    return x


class Network(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    self.feature_extractor = FeatureExtractor(cfg)
    self.classifier = Classifier(cfg)

    self.criterion = nn.CrossEntropyLoss()

  def forward(self, x, labels=None):
    feats = self.feature_extractor(x)
    logits = self.classifier(feats)
    out_dict = {'logits': logits}

    if self.training:
      loss = self.loss(logits, labels)
      out_dict['loss'] = loss

    return out_dict

  def loss(self, logits, labels):
    loss = self.criterion(logits, labels)
    return loss
