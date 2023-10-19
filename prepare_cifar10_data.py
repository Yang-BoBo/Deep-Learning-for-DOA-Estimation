import os
import json
import argparse
import pickle
import numpy as np
from PIL import Image

from config import Config

def unpickle(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict


def unpack_data(files, gt_dir, id):
  samples_per_file = 10000
  id_per_file = id % samples_per_file
  file_id = id // samples_per_file

  file_path = os.path.join(gt_dir, files[file_id])

  dict = unpickle(file_path)
  data = dict[b'data']
  labels = dict[b'labels']

  datum = data[id_per_file]
  datum_reshaped = np.reshape(datum, [3, 32, 32])
  label = labels[id_per_file]
  return datum_reshaped, label


def unpack_and_store_all(gt_dir, filenames, split):
  save_dir = os.path.join(gt_dir, split)
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  label_dict = dict()
  save_path_dict = os.path.join(gt_dir, "{}_labels.json".format(split))
  count = 0

  for filename in filenames:
    file_path = os.path.join(gt_dir, filename)
    data_dict = unpickle(file_path)
    data = data_dict[b'data']
    labels = data_dict[b'labels']
    for i, datum in enumerate(data):
      img_id = str(count + i).zfill(6)
      save_path = os.path.join(save_dir, img_id + '.png')
      datum_reshaped = np.reshape(datum, [3, 32, 32])
      datum_reshaped = np.transpose(datum_reshaped, [1, 2, 0])
      Image.fromarray(datum_reshaped).save(save_path)

      label = labels[i]
      label_dict[img_id] = label

    count += data.shape[0]

  with open(save_path_dict, 'w') as fp:
    json.dump(label_dict, fp)


def decode_and_store_cifar10(gt_dir):
  files_train = ['data_batch_{}'.format(i) for i in range(1, 6)]
  files_test = ['test_batch']

  unpack_and_store_all(gt_dir, files_train, 'train')
  unpack_and_store_all(gt_dir, files_test, 'test')


if __name__ == "__main__":
  print("Unpacking and storing the dataset files, this may take a minute...")
  cfg = Config()
  decode_and_store_cifar10(cfg.gt_dir)
  print("Preparing dataset completed.")
