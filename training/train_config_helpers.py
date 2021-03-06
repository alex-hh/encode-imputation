import os, json
from copy import deepcopy

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import models
from models.metrics import subset_mse, correlation_coefficient_loss, correlation_coefficient_r, correlation_coefficient_loss_rowwise, subset_corr
from models.losses import inv_transformed_mse
from utils.CONSTANTS import val_expt_names, train_expt_names, all_chromosomes, data_dir, output_dir, config_dir
from utils.callbacks import GeneratorVal, MovingAverageVal, MovingAverageCheckpoint, RunningLossLogger
from utils.full_data_loaders import TrainDataGeneratorHDF5, ValDataGeneratorHDF5, TestDataGeneratorHDF5
from utils.chunked_data_loaders import ChunkedTrainDataGeneratorHDF5


def save_train_config(config_name, config_dict, config_base):
  # json.dumps(config_dict)
  config_folder = 'experiment_settings/{}'.format(config_base)
  os.makedirs(config_folder, exist_ok=True)
  with open('{}/{}.json'.format(config_folder, config_name), 'w') as jsfile:
    json.dump(config_dict, jsfile, indent=2) # indent forces pretty printing

## train_config = {'model_kwargs':, 'train_kwargs', 'data_kwargs':}

def param_str(v):
  if type(v) in [list, tuple]:
    return '-'.join([str(o) for o in v])
  else:
    return str(v)

def save_experiment_params(base_config_dict, params, base_name):
  shorthand_sets = {'tk': 'train_kwargs', 'dk': 'data_kwargs',
                    'mk': 'model_kwargs', 'vk': 'val_kwargs'}
  for plist in params:
    config = deepcopy(base_config_dict)
    pv = []
    for s, param, v in plist:
      config[shorthand_sets.get(s, s)][param] = v
      print(param)
      if type(v) == dict:
        pv.append(param+'-'.join([str(k)+param_str(item) for k, item in v.items()]))
      else:
        pv.append(param+param_str(v))

    config['expt_name'] = base_name+'_'+'-'.join(pv)
    save_train_config('-'.join(pv), config, base_name)

def save_train_config(expt_set, expt_name, model_class, data_loader,
                      weighted_average=False, eval_freq=1000000,
                      train_kwargs={}):
  base_kwargs = {}
  base_kwargs['model_kwargs'] = model_class.config
  base_kwargs['model_kwargs']['model_class'] = model_class.__class__.__name__
  base_kwargs['data_kwargs'] = data_loader.config
  base_kwargs['data_kwargs']['data_class'] = data_loader.__class__.__name__
  base_kwargs['val_kwargs'] = {'weighted_average': weighted_average, 'eval_freq': eval_freq}
  base_kwargs['train_kwargs'] = train_kwargs

  config_output_path = config_dir + expt_set
  os.makedirs(config_output_path, exist_ok=True)
  with open(config_output_path + '/' + expt_name +'.json', 'w') as outf:
    json.dump(base_kwargs, outf, indent=2)

def get_validation_callbacks(val_model, val_gen, checkpoint_folder, expt_name, verbose=1,
                             eval_freq=1000000, weighted_average=False, test_run=False):
  checkpoint_path = checkpoint_folder + '/{}'.format(expt_name) + '_ep{epoch:02d}.{number}-{val_loss:.3f}.hdf5'
  callback_class = MovingAverageVal if weighted_average else GeneratorVal
  callbacks = [callback_class(val_gen, val_model, checkpoint_each_eval=True,
                              log_prefix='val_', verbose=verbose, eval_freq=500 if test_run else eval_freq, # TODO what does log prefix do?
                              checkpoint_path=checkpoint_path, max_checkpoints=2 if test_run else 50)]
  # Prints the cumulative / running loss every log_rate samples per epoch (sample counter resets to 0 each epoch) - just helps check for progress / verify script still running?
  callbacks.append(RunningLossLogger(log_rate=1 if test_run else 100000))
  return callbacks

def get_checkpoint_callbacks(checkpoint_folder, expt_name, weighted_average=False, verbose=1):
  checkpoint_path = checkpoint_folder + '/{}'.format(expt_name) +'_{epoch:02d}-{loss:.2f}.hdf5'
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
  # TODO - which of these config settings are relevant - test_weighted_average, test_time, secondary_chrom_size, custom_val, custom_splits, train_chrom_size, also_full_val, secondary_chroms
  # dk = "data_kwargs": {
  #   "data_class": "HDF5InMemDict",
  #   "n_drop": 50,
  #   "n_predict": 50,
  #   "dataset": "all",
  #   "directory": "/work/ahawkins/encodedata/",
  #   "chroms": [
  #     "chr13"
  #   ],
  #   "chunks_per_batch": 256,
  #   "transform": null,
  #   "custom_splits": false,
  #   "use_metaepochs": true,
  #   "subsample_each_epoch": true,
  #   "dataset_size": 1000000,
  #   "replace_gaps": true,
  #   "use_backup": false,
  #   "shuffle": true}
  train_gen = TrainDataGeneratorHDF5(**config['data_kwargs'])
  if train_only:
    return train_gen, None
  val_data_kwargs = deepcopy(config['data_kwargs'])
  val_data_kwargs.update(config['val_kwargs'])
  print(val_data_kwargs)
  val_gen = ValDataGeneratorHDF5(checkpoint_each_eval=True, **val_data_kwargs)
  # if config['val_kwargs'].get('test_weighted_average', False):
  #   print('Adding additional callback - shapshot val')
  #   secondary_val_gen = val_data_class(checkpoint_each_eval=False, log_prefix='snapshot_val_', **val_data_kwargs)
  #   val_gen = [val_gen, secondary_val_gen]
  if val_only:
    return None, val_gen
  return train_gen, val_gen