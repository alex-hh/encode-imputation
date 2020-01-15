import warnings, pickle

from keras import backend as K
from keras.optimizers import Adam

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