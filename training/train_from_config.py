import os
import argparse
import json
import numpy as np
from keras import backend as K
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard

from training.expt_config_loaders import load_models_from_config, load_data_from_config
from utils.callbacks import GeneratorVal, EpochTimer, RunningLossLogger,\
                            ResumableTensorBoard, MovingAverageVal, MovingAverageCheckpoint
from utils.CONSTANTS import BINNED_CHRSZ, data_dir, output_dir, config_dir, all_chromosomes

TEST_RUN = data_dir in ['data', 'data/', '/Users/alexhooker/projects/avocado/data', '/Users/alexhooker/projects/avocado/data/']

# TODO check this
def get_epoch_size(config):
  """
   Use settings to infer the number of samples (datapoints) per epoch
  """
  if 'dataset_size' in config['data_kwargs']:
    epoch_size = config['data_kwargs']['dataset_size']
    epochs = config['train_kwargs']['n_samples'] // epoch_size
  elif 'epochs' in config['train_kwargs']:
    epoch_size = BINNED_CHRSZ[config['data_kwargs']['chrom']], epochs = config['train_kwargs'].get('epochs', 10)
    # dataset_fraction = config['data_kwargs'].get('dataset_fraction', 1)
    # chroms = config['data_kwargs']['chroms']
    # if chroms == 'all':
    #   chroms = all_chromosomes
    # epoch_size = sum([dataset_fraction*BINNED_CHRSZ[chrom] for chrom in chroms])
  
  return epoch_size, epochs

def main(expt_name, expt_set):
  expt_name = expt_name.split('/')[-1]
  with open(os.path.join(config_dir, f'{expt_set}/{expt_name}.json'), 'r') as jsf:
    config = json.load(jsf)

  print('TEST_RUN', TEST_RUN, 'DATA_DIR', data_dir, 'OUTPUT_DIR', output_dir)
  # TODO figure out why I was previously working with list of val data
  train_gen, val_gen = load_data_from_config(config, local=TEST_RUN)
  train_model, val_model = load_models_from_config(config)

  epoch_size, epochs = get_epoch_size(config)
  
  weighted_average = config['train_kwargs'].get('weighted_average', False)
  learning_rate = K.eval(train_model.optimizer.lr)
  print(f'Learning rate: {learning_rate}, n epochs {epochs}, epoch size {epoch_size}')

  print(train_model.summary())
  
  checkpoint_folder = os.path.join(output_dir, f'weights/{expt_set}')
  os.makedirs(checkpoint_folder, exist_ok=True)

  epoch_checkpoint_path = checkpoint_folder + f'/{expt_name}' +'_{epoch:02d}-{loss:.2f}.hdf5'
  sample_checkpoint_path = checkpoint_folder + f'/{expt_name}' + '_ep{epoch:02d}.{number}-{val_loss:.3f}.hdf5'
  
  eval_freq = 1000000
  print('EVAL FREQ', eval_freq)

  callbacks = [EpochTimer()]
  
  checkpoint_each_eval = True
  print('CHECKPOINT EACH EVAL', checkpoint_each_eval)

  # Prepare validation callback - callback is required, because the val model and the train model
  #  are different (different numbers of outputs), so can't just supply train_gen and val_gen to fit_generator
  # N.B. if validation set is used, the validation callbacks handle checkpointing (checkpoint_each_eval controls this..),
  #  if there is no validation, checkpointing is handled by different callbacks
  if val_gen is not None:

    # TODO check/simplify these callbacks
    #  checkpointing config (inc max_checkpoints), metrics_to_keep 'all' config, 

    log_prefix = getattr(val_gen, 'log_prefix', 'val_')
    print('log prefix', log_prefix)
     
    if weighted_average and checkpoint_each_eval:
      callback_class = MovingAverageVal
    else:
      print("GENERATOR VAL FOR BASELINE (SNAPSHOT) VAL LOSS")
      callback_class = GeneratorVal

    callback_kwargs = {'eval_freq': 64 if TEST_RUN else eval_freq,
                       'checkpoint_path': sample_checkpoint_path,
                       'max_checkpoints': 2 if TEST_RUN else 7,
                       'verbose': 2 if TEST_RUN else 1}
    batch_val_callback = callback_class(val_gen, val_model, checkpoint_each_eval=checkpoint_each_eval,
                                        log_prefix=log_prefix, metrics_to_keep='all', **callback_kwargs)
    callbacks.append(batch_val_callback)
    # TODO check - what is this
    callbacks.append(RunningLossLogger(log_rate=1 if TEST_RUN else 100000))

  else:
    if weighted_average:
      callbacks.append(MovingAverageCheckpoint(checkpoint_path=epoch_checkpoint_path))
    else: 
      callbacks.append(ModelCheckpoint(epoch_checkpoint_path, monitor='loss'))

  start_epoch = 0
  if not TEST_RUN:
    callbacks += [CSVLogger(os.path.join(output_dir, f'logs/{expt_set}/{expt_name}.csv'), append=False), # append=True - if file exists allows continued training
                  # ModelCheckpoint(epoch_checkpoint_path, monitor='loss', period=1), # TODO - maybe drop this.. not really necessary
                  ResumableTensorBoard(start_epoch*epoch_size,
                                       log_dir=os.path.join(output_dir, f'logs/{expt_set}/{expt_name}/'), update_freq=100000)
                  ]

  train_model.fit_generator(train_gen, epochs=epochs, verbose=1 if TEST_RUN else 2,
                            callbacks=callbacks, initial_epoch=start_epoch)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('expt_name')
  parser.add_argument('expt_set')
  parser.add_argument('--seed', default=211, type=int)
  args = parser.parse_args()
  print('SETTING SEED', args.seed)
  np.random.seed(args.seed)
  main(args.expt_name, args.expt_set)