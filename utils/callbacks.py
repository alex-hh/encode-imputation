import os, glob, re, gc

from time import time
from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.callbacks import Callback, BaseLogger, TensorBoard
from keras.utils import Sequence
from keras.utils.data_utils import OrderedEnqueuer

from models.model_utils import save_optimizer, save

from utils.general import timeit


def is_sequence(seq):
    """Determine if an object follows the Sequence API.
    # Arguments
        seq: a possible Sequence object
    # Returns
        boolean, whether the object follows the Sequence API.
    """
    # TODO Dref360: Decide which pattern to follow. First needs a new TF Version.
    return (getattr(seq, 'use_sequence_api', False)
            or set(dir(Sequence())).issubset(set(dir(seq) + ['use_sequence_api'])))

class EpochTimer(Callback):
  # just make sure to include before csvlogger

  def on_epoch_begin(self, epoch, logs=None):
    self.epoch_time_start = time()

  def on_epoch_end(self, epoch, logs=None):
    epoch_time = time() - self.epoch_time_start
    logs['time'] = epoch_time

class ResumableTensorBoard(TensorBoard):
  # self.samples seen
  # self.samples seen at last write
  def __init__(self, samples_seen, samples_seen_at_last_write=None, **kwargs):
    super().__init__(**kwargs)
    if samples_seen_at_last_write is None:
      samples_seen_at_last_write = samples_seen
    self.samples_seen = samples_seen
    self.samples_seen_at_last_write = samples_seen_at_last_write

class RunningLossLogger(BaseLogger):
  """
   Extends base logger to allow printing the loss multiple times per epoch (every log_rate samples) instead of 
   once per epoch (what BaseLogger does).
   Sample counter resets to 0 each epoch.
  """

  def __init__(self, log_rate, **kwargs):
    self.log_rate = log_rate
    self.log_in = log_rate
    super().__init__(**kwargs)
    # print('INIT LOG CHECKPOINT')

  def on_batch_end(self, batch, logs=None):
    super().on_batch_end(batch, logs=logs) # this updates self.seen
    self.log_in -= logs['size']
    for k, v in self.totals.items(): # running totals
      if re.match('val', k):
        continue
      else:
        logs['running_{}'.format(k)] = v / self.seen
    # print('RunningLossCheckpointer: log in {} checkpoint in {} seen {}'.format(self.log_in, self.checkpoint_in, self.seen), flush=True)
    if self.log_in <= 0:
      print('Loss after {} samples:\t{}'.format(self.seen, logs['running_loss']), flush=True)
      self.log_in = self.log_rate

class BatchedValidationCallback(Callback):
  """
   Base callback for evaluating model on validation set (provided via data generator) and saving weights every eval_freq samples
   Child callbacks define an _evaluate_model method to actually compute desired validation metrics and update logs with them
  """

  def __init__(self, data_generator, val_model,
               eval_freq=None, checkpoint_each_eval=False,
               checkpoint_path=None, max_checkpoints=100, 
               log_prefix='val_', verbose=1):
    # main params
    self.data_generator = data_generator
    self.validation_steps = len(self.data_generator)
    self.val_model = val_model
    self.eval_freq = eval_freq
    self.checkpoint_each_eval = checkpoint_each_eval
    self.verbose = verbose
    self.checkpoint_path = checkpoint_path
    self.log_prefix = log_prefix # metric prefix (e.g. val_)
    # track existing checkpoints
    self.current_files = []
    self.max_checkpoints = max_checkpoints
    self.best_checkpoint = None
    # track overall progress
    self._samples_seen_since_last_saving = 0
    self._current_epoch = 0
    self._epoch_samples = 0
    self.eval_gen_workers = 1
    self._current_metrics = {}

  def on_epoch_begin(self, epoch, logs=None):
    self._epoch_samples = 0
    self._epoch_counter = 0
    self._current_epoch = epoch

  def on_batch_end(self, batch, logs=None):
    # print('Val on batch end')
    # https://github.com/tensorflow/tensorflow/blob/5912f51d580551e5cee2cfde4cb882594b4d3e60/tensorflow/python/keras/callbacks.py#L955
    logs = logs or {}
    logs.update(self._current_metrics)
    
    print(f'Batch end, samples seen {self._samples_seen_since_last_saving} eval freq {self.eval_freq}')

    if isinstance(self.eval_freq, int):
      self._samples_seen_since_last_saving += logs.get('size', 1)
      self._epoch_samples += logs.get('size', 1)

      if self._samples_seen_since_last_saving >= self.eval_freq:
        if self.verbose == 2:
          print('pre val callback logs', logs)
        self._evaluate_model(logs)
        if self.checkpoint_each_eval and self.checkpoint_path is not None:
          try:
            if self.verbose:
              print('Saving checkpoint epoch {} number {} val loss {}'.format(self._current_epoch,
                                                                              self._epoch_counter,
                                                                              logs['{}loss'.format(self.log_prefix)]))
          except:
            raise Exception('{} loss not in logs'.format(self.log_prefix), logs)
          if self.verbose == 2:
            print('checkpoint logs', logs)
          filepath = self.checkpoint_path.format(epoch=self._current_epoch,
                                                 number=self._epoch_counter,
                                                 val_loss=logs['{}loss'.format(self.log_prefix)])
          save(self.model, filepath)
          self.current_files.append('.'.join(filepath.split('.')[:-1]))
          self.cleanup()
          print('gcing')
          gc.collect()
        self._samples_seen_since_last_saving = 0
        self._epoch_counter += 1

  def _evaluate_model(self, logs):
    raise NotImplementedError()

  def on_epoch_end(self, epoch, logs=None):
    logs.update(self._current_metrics)
    if self.verbose==2:
      print('End of epoch logs', logs)
    if self.eval_freq is None:
      self._evaluate_model(logs)

  def cleanup(self):
    if len(self.current_files) > self.max_checkpoints:
      to_delete = glob.glob(self.current_files.pop(0)+'*')
      for f in to_delete:
        if self.verbose:
          print('Discarding old file {}'.format(f))
        os.remove(f)

class GeneratorVal(BatchedValidationCallback):
  """
  Subclasses batched validation callback to retain the checkpointing every eval_freq samples functionality
  But instead of manually computing the loss in a batched way on the validation set use keras evaluate_generator
  to evaluate all of the metrics associated with the compiled val model.

  Uses keras orderedenqueuer to try to facilitate parallelisation, but due to difficulties only use single worker for now
  """
  def __init__(self, data_generator, val_model, metrics_to_keep='all', **kwargs):
    self.metrics_to_keep = metrics_to_keep # optionally filter the metrics to track
    super().__init__(data_generator, val_model, **kwargs)
    assert is_sequence(self.data_generator), 'validation generator must be an instance of keras.utils.Sequence'
    val_enqueuer = OrderedEnqueuer(
        self.data_generator,
        use_multiprocessing=False)
    # n.b. that the enqueuer calls on_epoch_end: https://github.com/keras-team/keras/blob/efe72ef433852b1d7d54f283efff53085ec4f756/keras/utils/data_utils.py
    val_enqueuer.start(workers=1,
                       max_queue_size=10)
    self.data_generator = val_enqueuer.get()
    self.eval_gen_workers = 0

  def _evaluate_model(self, logs=None):
    if self.verbose:
      print('evaluating model (generator val)', flush=True)

    metric_dict = {}
    metrics = self.val_model.evaluate_generator(self.data_generator, self.validation_steps,
                                                workers=0, verbose=self.verbose) # workers = 0 here is because the enqueuer handles stuff
    if type(metrics) == list:
      for metric, score in zip(self.val_model.metrics_names, metrics):
        if self.metrics_to_keep == 'all' or metric in self.metrics_to_keep:
          metric_dict[metric] = score
    else:
      # if there is only a single metric, assume it is the loss
      metric_dict['loss'] = metrics

    for k, v in metric_dict.items():
      if self.verbose:
        print('epoch {} {}{} after {} samples:\t{}'.format(self._current_epoch, self.log_prefix, k, self._epoch_samples, v), flush=True)
      if logs is not None:
        logs['{}{}'.format(self.log_prefix, k)] = v
      self._current_metrics['{}{}'.format(self.log_prefix, k)] = v
    if hasattr(self.data_generator, 'release_memory'):
      self.data_generator.release_memory()

class MovingAverageValMixin:
  """
   Mixin to provide evaluation and checkpointing functionality for exponentially weighted average of model weights
   Based on the callback here https://github.com/alno/kaggle-allstate-claims-severity/blob/master/keras_util.py
  """

  def __init__(self, *args, decay=0.998, save_mv_ave_model=True, 
               custom_objects={}, verbose=1, **kwargs):
    super().__init__(*args, **kwargs)
    self.decay = decay
    self.custom_objects = custom_objects  # dictionary of custom layers
    self.sym_trainable_weights = None  # trainable weights of model
    self.mv_trainable_weights_vals = None  # moving averaged values
    self.verbose = verbose

  def on_train_begin(self, logs={}):
    self.sym_trainable_weights = self.model.trainable_weights
    # Initialize moving averaged weights using original model values
    self.mv_trainable_weights_vals = {x.name: K.get_value(x) for x in
                                      self.sym_trainable_weights}
    if self.verbose:
      print('Created a copy of model weights to initialize moving'
            ' averaged weights.')
    super().on_train_begin(logs=logs)

  def _evaluate_model(self, logs=None, **kwargs):
    # 1. store weights
    current_weights = []
    # current_weights = [K.get_value(x) for x in self.sym_trainable_weights]
    if self.verbose:
      print('setting ewa weights for validation')
    for weight in self.sym_trainable_weights:
      current_weights.append(K.get_value(weight))
      K.set_value(weight, self.mv_trainable_weights_vals[weight.name])
    super()._evaluate_model(logs=logs, **kwargs)
    if self.checkpoint_each_eval:
      self._save_mva_model(logs=logs) # actual model will also be saved by parent callback
    if self.verbose:
      print('resetting weights')
    for weight, _curr_value in zip(self.sym_trainable_weights, current_weights):
      K.set_value(weight, _curr_value)

  def on_batch_end(self, *args, **kwargs):
    for weight in self.sym_trainable_weights:
      old_val = self.mv_trainable_weights_vals[weight.name]
      self.mv_trainable_weights_vals[weight.name] -= \
          (1.0 - self.decay) * (old_val - K.get_value(weight))
    super().on_batch_end(*args, **kwargs)

  def _save_mva_model(self, logs):
    filepath = self.checkpoint_path.format(epoch=self._current_epoch,
                                           number=str(self._epoch_counter)+'-ewa',
                                           val_loss=logs['{}loss'.format(self.log_prefix)])
    self.model.save_weights(filepath)

class MovingAverageVal(MovingAverageValMixin, GeneratorVal):
  """
  Subclasses generator val to evaluate and save exponentially weighted average of model weights every eval_freq samples
  """
  pass

# TODO -- IMPLEMENT
class MovingAverageCheckpoint(MovingAverageValMixin, Callback):

  def __init__(self, checkpoint_path=None, decay=0.998):
    self.checkpoint_path = checkpoint_path
    self.decay = decay
    # super().__init__(*args)

  def on_epoch_end(self, epoch, logs=None):
    raise NotImplementedError()
  #   filepath = self.checkpoint_path.format(epoch=epoch,
  #                                          loss=logs['loss'])
  #   self.model.save_weights(filepath)