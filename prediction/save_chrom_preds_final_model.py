import glob, sys, os, re, argparse, json
from itertools import islice

import h5py
import numpy as np
import pandas as pd

from models import DAE, cauchy5
from models.model_utils import find_checkpoint
from utils.full_data_loaders import TestDataGeneratorHDF5, ValDataGeneratorHDF5
from utils.CONSTANTS import dataset_expts, BINNED_CHRSZ, data_dir, output_dir


def main(model_name, chrom, expt_set=None, checkpoint_code=None, outfmt='npz', dataset='test', 
         train_dataset='all', moving_average=False, output_directory=None, data_directory=None,
         checkpoint_file=None):
  if expt_set in ['imp', 'imp1']:
    checkpoint_code = 14 if expt_set == 'imp' else 14.0 # these are just used to identify the weights file that is loaded
  if output_directory is None:
    output_directory = output_dir
  if data_directory is None:
    data_directory = data_dir

  print('Saving preds for {} on {} to {}'.format(model_name, dataset, output_dir))
  assert train_dataset in ['train', 'all'], 'train dataset must be either train or all'

  if dataset == 'test':
    data_gen = TestDataGeneratorHDF5(train_dataset=train_dataset, n_drop=50,
                                     chrom=chrom, directory=data_directory)
  elif dataset == 'val':
    if train_dataset == 'all':
      raise NotImplementedError()
    data_gen = ValDataGeneratorHDF5(n_drop=50, chrom=chrom, directory=data_directory)

  else:
    raise Exception('dataset must be either train or val')

  n_predict = len(dataset_expts[dataset])
  model = DAE(obs_counts=[50, n_predict], n_drop=50, mlp_dropout=0.3,
              dae_dim=[100,50], n_train_obs=len(dataset_expts[train_dataset]))
  pred_model = model.models[n_predict]
  pred_model.compile(loss='mse', optimizer='adam')

  checkpoint = checkpoint_file or find_checkpoint(model_name, expt_set=expt_set, checkpoint_code=checkpoint_code, moving_avg=moving_average,
                                                  weights_dir=output_directory)
  print('Loading checkpoint', checkpoint)
  pred_model.load_weights(checkpoint)

  print('Making predictions')
  preds = pred_model.predict_generator(data_gen, verbose=1, steps=None)
  print('Pred shape', preds.shape)
  preds = np.squeeze(preds)
  print('Squeezed pred shape', preds.shape)

  imp_dir = os.path.join(output_directory, f'{dataset}_imputations', '' if expt_set is None else expt_set, model_name)
  os.makedirs(imp_dir, exist_ok=True)
  
  print('Saving preds')
  checkpoint_str = '' if checkpoint_code is None else '.' + str(checkpoint_code)
  assert n_predict == preds.shape[1], 'check names - length of name list doesnt match data'
  for track_name, track_vals in zip(dataset_expts[dataset], preds.T):
    np.savez_compressed(os.path.join(imp_dir, '{}.{}{}.npz'.format(track_name, chrom, checkpoint_str)), track_vals.reshape(-1))

  print('Done')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('model_name') # name of specific weights file (e.g. chromschr1_14-0.48.hdf5)
  parser.add_argument('chrom') # name of chromsome e.g. chr1
  parser.add_argument('--dataset', default='test')
  parser.add_argument('--expt_set', type=str, default=None)
  parser.add_argument('--train_dataset', default='all')
  parser.add_argument('--moving_average', action='store_true')
  parser.add_argument('--checkpoint_code', default='14', type=str) # identifies checkpoint, should be 14 for imp and 14.0 for imp1
  parser.add_argument('--data_directory', type=str, default=None)
  parser.add_argument('--output_directory', type=str, default=None)
  parser.add_argument('--checkpoint_file', type=str, default=None)
  args = parser.parse_args()
  main(args.model_name, args.chrom, expt_set=args.expt_set, checkpoint_code=args.checkpoint_code,
       dataset=args.dataset, train_dataset=args.train_dataset, moving_average=args.moving_average,
       data_directory=args.data_directory, output_directory=args.output_directory,
       checkpoint_file=args.checkpoint_file)