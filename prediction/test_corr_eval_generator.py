import glob, sys, os, re, argparse, json
from itertools import islice

import h5py
import numpy as np
import pandas as pd

from utils.full_data_loaders import TestDataGeneratorHDF5, ValDataGeneratorHDF5
from utils.CONSTANTS import data_dir, output_dir, config_dir, dataset_expts, BINNED_CHRSZ, n_val_expts
from models.metrics import PearsonCorrelation
from training.expt_config_loaders import load_models_from_config


# TEST_RUN = data_dir in ['data/', 'data']
TEST_RUN = False

def find_checkpoint(model_name, expt_set, checkpoint_code, moving_avg=False):
  # find the path of the desired weight file
  checkpoint_path = os.path.join(output_dir, 'weights/{}/{}_ep{}{}*.hdf5'.format(expt_set, model_name, checkpoint_code, '-ewa' if moving_avg else ''))
  checkpoints = glob.glob(checkpoint_path)
  assert len(checkpoints) == 1, 'check checkpoints : found {} : {}'.format(len(checkpoints), checkpoints, checkpoint_path)
  return checkpoints[0]

def get_train_track_names(train_dataset='all'):
  if train_dataset == 'train':
    return dataset_expts['train']
  elif train_dataset == 'all':
    return dataset_expts['train'] + dataset_expts['val']
  else:
    raise Exception('Train dataset must be either train or all')

def main(model_name, expt_set, chrom, checkpoint_code, outfmt='npz', dataset='val'):
  model_name = model_name.split('/')[-1]
  print('Saving preds for {} on {} to {}'.format(model_name, dataset, output_dir))
  
  print('model name', model_name, 'expt_set', expt_set, 'chrom', chrom, 'checkpoint_code', checkpoint_code)
  n_test_experiments = 0

  config_file = os.path.join(config_dir, '{}/{}.json'.format(expt_set, model_name))
  with open(config_file, 'r') as jf:
    config = json.load(jf)

  train_dataset = config['data_kwargs'].get('train_dataset', 'train')
  assert train_dataset in ['train', 'all'], 'train dataset must be either train or all'
  train_track_names = get_train_track_names(train_dataset)
  val_track_names = dataset_expts['val']
  test_track_names = dataset_expts['test']
  n_drop = config['data_kwargs']['n_drop']
  msg = 'Model was trained on {} dataset to reconstruct {} track values from remaining {}'.format(
            'train' if train_dataset == 'train' else 'full (train+val)',
            n_drop, len(train_track_names)-n_drop)

  moving_average = config['val_kwargs'].get('weighted_average', False)
  if moving_average:
    msg += '.  Exponentially weighted average of weights used for predictions.'
  print(msg)

  if dataset == 'test':
    pred_track_names = test_track_names
    data_gen = TestDataGeneratorHDF5(train_dataset=train_dataset, n_drop=n_drop,
                                     chrom=chrom, directory=data_dir)
  elif dataset == 'val':
    pred_track_names = val_track_names
    if train_dataset == 'all':
      raise NotImplementedError()
    data_gen = ValDataGeneratorHDF5(n_drop=n_drop, chrom=chrom, directory=data_dir)

  else:
    raise Exception('dataset must be either train or val')

  # TODO: add additional metrics
  _, model = load_models_from_config(config_file, val_n_tracks=len(pred_track_names), extra_metrics=[PearsonCorrelation()])
  checkpoint = find_checkpoint(model_name, expt_set, checkpoint_code, moving_avg=moving_average)
  print('Loading checkpoint', checkpoint)
  model.load_weights(checkpoint)

  # print('Making predictions')
  corr = model.evaluate_generator(data_gen, verbose=1, steps=5 if TEST_RUN else None)
  print(corr)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('model_name')
  parser.add_argument('expt_set')
  parser.add_argument('chrom')
  parser.add_argument('checkpoint_code') # ep07.1
  parser.add_argument('dataset')
  args = parser.parse_args()
  main(args.model_name, args.expt_set, args.chrom, args.checkpoint_code, dataset=args.dataset)