import torch
from torch.utils.data import DataLoader
import torch
from config import Config
from data import Cifar10Dataset

from model import Network


def test():
  # Configuration settings
  cfg = Config()

  # Load dataset
  dataset = Cifar10Dataset(cfg.gt_dir, split='test')
  dataloader = DataLoader(dataset, batch_size=cfg.batch_size_test, shuffle=True, num_workers=4)

  # Initialize network
  model = Network(cfg)
  model.eval()
  if cfg.enable_cuda:
    model = model.cuda()

  # Load model from saved file
  saved_model_path = 'model.pth'
  print("Loading model from {}...".format(saved_model_path))
  model.load_state_dict(torch.load('model.pth'))

  correct = 0
  total = 0
  print("Starting evaluation...")
  with torch.no_grad():
    for data in dataloader:
      images, labels = data
      if cfg.enable_cuda:
        images, labels = images.cuda(), labels.cuda()
      # Feed images through model and generate outputs
      outputs = model(images)
      logits = outputs['logits']
      # The class with the highest score is the prediction
      _, predicted = torch.max(logits.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  print("Accuracy of the network on the {} test images: {}".format(total, round((correct/total*100), 1)))
  print("Correct predictions: {}".format(correct))
  print("Total predictions: {}".format(total))


if __name__ == "__main__":
  test()
