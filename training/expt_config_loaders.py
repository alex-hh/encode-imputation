import os, json

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import models
from models.metrics import subset_mse, correlation_coefficient_loss, correlation_coefficient_r, correlation_coefficient_loss_rowwise, subset_corr
from models.losses import inv_transformed_mse
from utils.CONSTANTS import val_expt_names, train_expt_names, all_chromosomes, data_dir, output_dir, config_dir
from utils.callbacks import GeneratorVal, MovingAverageVal, MovingAverageCheckpoint, RunningLossLogger
from utils.full_data_loaders import TrainDataGeneratorHDF5, ValDataGeneratorHDF5, TestDataGeneratorHDF5
from utils.chunked_data_loaders import ChunkedTrainDataGeneratorHDF5


def get_validation_callbacks(val_model, val_gen, checkpoint_folder, expt_name, verbose=1,
                             eval_freq=1000000, weighted_average=False, test_run=False):
  checkpoint_path = os.path.join(checkpoint_folder, '{}'.format(expt_name) + '_ep{epoch:02d}.{number}-{val_loss:.3f}.hdf5')
  callback_class = MovingAverageVal if weighted_average else GeneratorVal
  callbacks = [callback_class(val_gen, val_model, checkpoint_each_eval=True,
                              log_prefix='val_', verbose=verbose, eval_freq=500 if test_run else eval_freq, # TODO what does log prefix do?
                              checkpoint_path=checkpoint_path, max_checkpoints=2 if test_run else 50)]
  # Prints the cumulative / running loss every log_rate samples per epoch (sample counter resets to 0 each epoch) - just helps check for progress / verify script still running?
  callbacks.append(RunningLossLogger(log_rate=1 if test_run else 100000))
  return callbacks

def get_checkpoint_callbacks(checkpoint_folder, expt_name, weighted_average=False, verbose=1):
  checkpoint_path = os.path.join(checkpoint_folder, '{}'.format(expt_name) +'_{epoch:02d}-{loss:.2f}.hdf5')
  if weighted_average:
    raise NotImplementedError('MovingAverageCheckpoint doesnt do anything rn as far as I can tell')
    # callbacks = [MovingAverageCheckpoint(checkpoint_path=epoch_checkpoint_path, verbose=verbose)]
  else: 
    callbacks = [ModelCheckpoint(epoch_checkpoint_path, monitor='loss', verbose=verbose)]
  # Prints the cumulative / running loss every log_rate samples per epoch (sample counter resets to 0 each epoch)
  callbacks.append(RunningLossLogger(log_rate=1 if test_run else 100000))
  return callbacks

def load_models_from_config(config, val_n_tracks=None):
  """
   loads model and loss objects based on names in config dict and returns compiled training and prediction models
      training model output size is the number of tracks used in training (as defined in config) (if n_predict/n_drop in data_kwargs)
      prediction model output size is the number of tracks used at test time (defined in config, but flexible via val_n_tracks)
  """
  if type(config) == str:
    with open(config, 'r') as jf:
      config = json.load(jf)
  model_class_name = config['model_kwargs'].pop('model_class')
  
  # update number of predictions required at test time
  if val_n_tracks is not None:
    config['model_kwargs']['obs_counts'][-1] = val_n_tracks

  # build model
  model = getattr(models, model_class_name)(**config['model_kwargs'])
  n_predict = config['data_kwargs'].get('n_predict', config['data_kwargs']['n_drop'])  
  if n_predict is None:
    n_predict = config['data_kwargs']['n_drop']
  train_model = model.models[n_predict]
  val_model = model.models[config['model_kwargs']['obs_counts'][-1]]

  # load loss functions from models module from function names
  train_loss = getattr(models, config['train_kwargs']['loss'])
  val_loss = getattr(models, config['val_kwargs'].get('loss', 'mse'))

  val_metrics = []
  monitor_all = config['train_kwargs'].get('monitor_all', False)
  val_all_expt_metrics = [subset_mse([i], e) for i, e in enumerate(val_expt_names)]

  if monitor_all:
    val_metrics += val_all_expt_metrics
  
  train_model.compile(loss=train_loss, optimizer=Adam(config['train_kwargs']['lr']))
  val_model.compile(loss=val_loss, optimizer='adam', metrics=val_metrics)
  return train_model, val_model

def load_data_from_config(config, local=False, val_only=False, custom_kwargs={}, test_time=False, train_only=False):
  """
   val_only: at test time to generate predictions (loading from config)
   train_only: when training on full (train+val) dataset set to true
   returns a train data generator and a val data generator
  """

  data_class = config['data_kwargs'].pop('data_class')
  assert data_class in ['HDF5InMemDict', 'TrainDataGeneratorHDF5', 'ChunkedTrainDataGeneratorHDF5']
  if ('directory' not in config['data_kwargs']) or local:
    config['data_kwargs']['directory'] = data_dir
  data_directory = config['data_kwargs']['directory']
  
  if data_class == 'HDF5InMemDict':
    from utils.old_data_loaders import HDF5InMemDict
    train_gen = HDF5InMemDict(**config['data_kwargs'])
  elif data_class == 'TrainDataGeneratorHDF5':
    train_gen = TrainDataGeneratorHDF5(**config['data_kwargs'])
  elif data_class == 'ChunkedTrainDataGeneratorHDF5':
    train_gen = ChunkedTrainDataGeneratorHDF5(**config['data_kwargs'])
  # train_gen = TrainDataGeneratorHDF5(**config['data_kwargs'])
  if train_only:
    print(f"Train only, class {data_class}, dir {data_directory}")
    return train_gen, None
  
  # HARDCODE VAL KWARGS ---> ONLY EVALUATE ON CHR21
  val_gen = ValDataGeneratorHDF5(batch_size=256, n_drop=50, directory=data_directory, train_dataset='train',
                                 chrom='chr21', replace_gaps=config['data_kwargs'].get('replace_gaps', True))

  if val_only:
    print(f"Val only, class ValDataGeneratorHDF5, dir {data_directory}")
    return None, val_gen
  
  print(f"Train class {data_class}, val class ValDataGeneratorHDF5, dir {data_directory}")
  return train_gen, val_gen