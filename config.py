from dataclasses import dataclass


@dataclass
class Config:
    batch_size_train = 32
    batch_size_test = 25
    lr = 0.005
    lr_momentum = 0.9
    weight_decay = 1e-4
    num_classes = 10
    gt_dir = "./data/cifar-10-batches-py/"
    num_iterations = 10000
    log_iterations = 100
    enable_cuda = False
