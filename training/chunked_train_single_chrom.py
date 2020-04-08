import os
import argparse

import numpy as np

from keras import backend as K
from keras.callbacks import CSVLogger
from keras.optimizers import Adam

from models import DAE, cauchy5
from utils.callbacks import EpochTimer, ResumableTensorBoard, MovingAverageVal, MovingAverageCheckpoint
from training.expt_config_loaders import get_checkpoint_callbacks, get_validation_callbacks
from utils.chunked_data_loaders import ChunkedTrainDataGeneratorHDF5
from utils.full_data_loaders import ValDataGeneratorHDF5
from utils.CONSTANTS import data_dir, output_dir


def main(train_dataset, expt_set=None, model_name=None, chrom='chr21', test_run=False,
         weighted_average=False, save_logs=False, eval_freq=1000000, epochs=20,
         epoch_size=1000000, seed=211, data_directory=None, output_directory=None):
  if data_directory is None:
    data_directory = data_dir
  if output_directory is None:
    output_directory = output_dir
  if model_name is None:
    model_name = '{}_{}'.format(chrom, train_dataset)
  if train_dataset=='full':
    n_train_obs = 312
  elif train_dataset == 'train':
    n_train_obs = 267
  else: 
    raise ValueError('Train set must be either full or train')

  np.random.seed(seed)
  # n_drop and obs_counts are decoupled to allow the possibility for training on a subset of the dropped signals
  model = DAE(obs_counts=[50,45], n_drop=50, mlp_dropout=0.3,
              dae_dim=[100,50], n_train_obs=n_train_obs)

  # TODO fix naming (model.models[50] -> model.models['train']?)
  train_model = model.models[50]
  train_model.compile(loss=cauchy5, optimizer=Adam(lr=0.0003))

  train_gen = ChunkedTrainDataGeneratorHDF5(n_drop=50, chrom=chrom, batch_size=256,
                                            epoch_size=epoch_size, replace_gaps=True,
                                            directory=data_directory)
  # from training.expt_config_savers import save_train_config
  # save_train_config(expt_set, model_name, model, train_gen,
  #                   weighted_average=weighted_average, eval_freq=eval_freq,
  #                   train_kwargs={'epochs': epochs, 'dataset_size': dataset_size,
  #                                 'loss': 'cauchy5', 'optimizer': 'adam',
  #                                 'lr': 0.0003, 'seed': seed})

  checkpoint_folder = os.path.join(output_directory, 'weights', '' if expt_set is None else expt_set)
  os.makedirs(checkpoint_folder, exist_ok=True)
  callbacks = [EpochTimer()] # does what it sounds like

  if train_dataset == 'train':
    # callbacks monitor metrics on val set (for training chromosome) as well as saving checkpoints
    val_model = model.models[45]
    val_model.compile(loss='mse', optimizer=Adam())
    val_gen = ValDataGeneratorHDF5(train_dataset=train_dataset, chrom=chrom, batch_size=256,
                                   directory=data_directory)
    callbacks += get_validation_callbacks(val_model, val_gen, checkpoint_folder, model_name,
                                          weighted_average=weighted_average, eval_freq=dataset_size,
                                          test_run=test_run, verbose=2 if test_run else 1)
  else:
    # callbacks save weights each dataset_size samples (i.e. each 'epoch')
    callbacks += get_checkpoint_callbacks(checkpoint_folder, model_name, weighted_average=weighted_average)

  if save_logs and not test_run:
    callbacks += [CSVLogger(os.path.join(output_directory, 'logs', '' if expt_set is None else expt_set, '{}.csv'.format(model_name)), append=False),
                  ResumableTensorBoard(0,
                                       log_dir=os.path.join(output_directory, 'logs', '' if expt_set is None else expt_set, model_name),
                                       update_freq=100000)
                  ]

  train_model.fit_generator(train_gen, epochs=epochs, verbose=1 if test_run else 2,
                            callbacks=callbacks)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--expt_set', type=str, default=None)
  parser.add_argument('--train_dataset', default='train')
  parser.add_argument('--eval_freq', type=int, default=1000000)
  parser.add_argument('--epoch_size', type=int, default=1000000)
  parser.add_argument('--epochs', type=int, default=14)
  parser.add_argument('--model_name', default=None)
  parser.add_argument('--chrom', default='chr21')
  parser.add_argument('--test_run', action='store_true')
  parser.add_argument('--weighted_average', action='store_true')
  parser.add_argument('--save_logs', action='store_true')
  parser.add_argument('--seed', default=211, type=int)
  parser.add_argument('--data_directory', type=str, default=None)
  parser.add_argument('--output_directory', type=str, default=None)
  args = parser.parse_args()
  print(args)
  main(args.train_dataset, expt_set=args.expt_set, model_name=args.model_name,
       chrom=args.chrom, test_run=args.test_run, weighted_average=args.weighted_average,
       save_logs=args.save_logs, eval_freq=args.eval_freq, epochs=args.epochs,
       seed=args.seed, data_directory=args.data_directory, output_directory=args.output_directory,
       epoch_size=epoch_size)