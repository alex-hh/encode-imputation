import os, glob, re, warnings, pickle

from keras import backend as K
from keras.optimizers import Adam

from utils.CONSTANTS import output_dir


def find_last_checkpoint(model_name, expt_set, weights_dir, moving_avg=False):
  print('Searching for checkpoints in dir', weights_dir)
  checkpoint_path = os.path.join(weights_dir, 'weights', '' if expt_set is None else expt_set,
                                 '{}_ep*{}.hdf5'.format(model_name, '-ewa*' if moving_avg else ''))
  print('Searching for checkpoints matching', checkpoint_path)
  checkpoints = glob.glob(checkpoint_path)
  last_checkpoint_path = None
  last_checkpoint_num = 0
  for checkpoint in checkpoints:
    try:
      checkpoint_code = float(re.search('\d+.?\d+', checkpoint).group(0))
      if checkpoint_code > last_checkpoint_num:
        last_checkpoint_path = checkpoint
    except:
      pass
  if last_checkpoint_path is None:
    raise Exception('No matching checkpoints found')
  return last_checkpoint_path

def find_checkpoint(model_name, expt_set=None, checkpoint_code=None, moving_avg=False, weights_dir=None):
  # find the path of the desired weight file
  if weights_dir is None:
    weights_dir = output_dir
  if checkpoint_code is None:
    return find_last_checkpoint(model_name, expt_set, weights_dir, moving_avg=moving_avg)
  print('Searching for checkpoints in dir', weights_dir)
  checkpoint_path = os.path.join(output_dir, 'weights', '' if expt_set is None else expt_set,
                                 '{}_ep{}{}*.hdf5'.format(model_name, checkpoint_code, '-ewa' if moving_avg else ''))
  print('Searching for checkpoints matching', checkpoint_path)
  checkpoints = glob.glob(checkpoint_path)
  assert len(checkpoints) == 1, 'check checkpoints : found {} : {}'.format(len(checkpoints), checkpoints, checkpoint_path)
  return checkpoints[0]

def save_optimizer(model, checkpoint_file):
  # https://github.com/keras-team/keras/blob/efe72ef433852b1d7d54f283efff53085ec4f756/keras/engine/saving.py#L146
  training_config = {'optimizer_config': {
                     'class_name': model.optimizer.__class__.__name__,
                     'config': model.optimizer.get_config()},
                     'loss': model.loss,
                     'metrics': model.metrics,
                     'weighted_metrics': model.weighted_metrics,
                     'sample_weight_mode': model.sample_weight_mode,
                     'loss_weights': model.loss_weights # for model with multiple losses
                     }
  symbolic_weights = getattr(model.optimizer, 'weights')
  if symbolic_weights:
    weight_values = K.batch_get_value(symbolic_weights)
    training_config['optimizer_weights'] = weight_values
  with open(checkpoint_file, 'wb') as f:
    pickle.dump(training_config, f)

def save(model, filepath):
  model.save_weights(filepath)
  save_optimizer(model, '.'.join(filepath.split('.')[:-1])+'_optimizer.pkl')

def load_optimizer_weights(model, checkpoint_file):
  with open(checkpoint_file, 'rb') as f:
    training_config = pickle.load(f)
  if 'optimizer_weights' in training_config:
    model._make_train_function()
    print('Loading optimizer weights')
    try:
      model.optimizer.set_weights(training_config['optimizer_weights'])
    except ValueError:
      warnings.warn('Error in loading the saved optimizer '
                    'state. As a result, your model is '
                    'starting with a freshly initialized '
                    'optimizer.')
  else:
    print('No optimizer weights saved')
  return model

def load_optimizer_and_compile(model, checkpoint_file):
  # to be applied to a model which has already been compiled
  # https://github.com/keras-team/keras/blob/efe72ef433852b1d7d54f283efff53085ec4f756/keras/engine/saving.py#L314
  with open(checkpoint_file, 'rb') as f:
    training_config = pickle.load(f)

  assert training_config['optimizer_config']['class_name'] == 'Adam'
  optimizer = Adam(**training_config['optimizer_config']['config'])
  model.compile(optimizer=optimizer, loss=training_config['optimizer_config']['loss'],
                metrics=training_config['optimizer_config']['metrics'])

  if 'optimizer_weights' in training_config:
    model._make_train_function()
    print('Loading optimizer weights')
    try:
      model.optimizer.set_weights(training_config['optimizer_weights'])
    except ValueError:
      warnings.warn('Error in loading the saved optimizer '
                    'state. As a result, your model is '
                    'starting with a freshly initialized '
                    'optimizer.')
  else:
    print('No optimizer weights saved')
  return model