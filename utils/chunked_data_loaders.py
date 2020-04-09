import math, gc, os
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd

from keras.utils import Sequence

from utils.CONSTANTS import all_chromosomes, BINNED_CHRSZ, CHRSZ, train_cell2id,\
                            train_assay2id, data_dir, dataset_expts


class BaseChunkedDataGeneratorHDF5(Sequence):
  """
  Load fixed number of datapoints into memory at each 'epoch', respecting chunking in h5py file to ensure good read performance
  Iterate over full chromosome in multiple epochs
  Shuffle (chunks rather than datapoints) at end of one full pass through chromosome 

   epoch_size: if None, load full dataset into memory
  """
  track_resolution = 25
  # TODO implement a simplified version of the data generator actually used for training
  def __init__(self, epoch_size=1000000, replace_gaps=True,
               batch_size=256, n_drop=50, dataset='train',
               chrom='chr21', measurements_per_chunk=100,
               constant_size=True, directory=None, n_predict=None,
               shuffle=True, debug=False):
    self.config = locals()
    del self.config['self']
    if directory is None:
      directory = data_dir
    if n_predict is None:
      n_predict = n_drop
    self.directory = directory
    self.n_predict = n_predict
    self.n_drop = n_drop
    self.batch_size = batch_size
    self.dataset = dataset
    self.epoch_size = epoch_size
    self.replace_gaps = replace_gaps
    self.measurements_per_chunk = measurements_per_chunk
    self.constant_size = constant_size
    self.chrom = chrom
    self.shuffle = shuffle
    self.debug = debug

    if epoch_size is None:
      self.epoch_size = BINNED_CHRSZ[chrom]
    self.total_chrom_chunks = math.ceil(BINNED_CHRSZ[chrom]/self.measurements_per_chunk)
    self.set_chunk_indexes()
    self.on_epoch_end()

  def get_chunk_startbin_from_index(self, chunk_index):
    if chunk_index > self.total_chrom_chunks:
      raise ValueError('Chunk index overflow')
    chunk_startbin = self.measurements_per_chunk * chunk_index
    return self.chrom, chunk_startbin

  def load_gap_info(self):
    self.gaps = pd.read_csv('{}gap.txt'.format(data_dir), sep='\t',
                            names=['chromosome', 'start', 'end', 'chr_id',
                                   'gap_char','gap_len', 'gap_type', '-'])
    self.bins_with_gaps = []
    chunk_indexes = []
    chrom_gaps = self.gaps[self.gaps['chromosome']==self.chrom]
    print(len(chrom_gaps))
    end_chunk = 0

    for ix, row in chrom_gaps.iterrows():
      start_bin = row['start'] // self.track_resolution
      start_chunk = start_bin // self.measurements_per_chunk
      chunk_indexes += list(range(end_chunk, min(start_chunk, BINNED_CHRSZ[self.chrom])))
      end_bin = row['end'] // self.track_resolution
      end_chunk = end_bin // self.measurements_per_chunk

      if self.debug:
        print('Adding gaps', row['start'], row['end'], start_bin, end_bin, start_chunk, end_chunk)
      self.bins_with_gaps += [b for b in range(start_bin, end_bin+1)]
    
    print('Chunks without gaps', len(chunk_indexes))
    return np.asarray(chunk_indexes)

  def set_chunk_indexes(self):
    """
     sets self.chunk_indexes
          self.n_epoch_chunks (number of chunks per epoch)
    """
    if self.replace_gaps:
      print('loading gap info')
      chunk_indexes = self.load_gap_info()
      self.chunk_indexes = np.asarray([ci for ci in chunk_indexes if ci < self.total_chrom_chunks])
    else:
      self.chunk_indexes = np.arange(self.total_chrom_chunks)
    
    self.n_epoch_chunks = min(self.epoch_size // self.measurements_per_chunk, len(self.chunk_indexes))
    print('Total number of chunks to be used in {}: {} of {} possible ungapped chunks, of {} possible total chunks'.format(self.dataset, self.n_epoch_chunks,
          len(self.chunk_indexes), self.total_chrom_chunks))

    if self.epoch_size < (self.total_chrom_chunks * self.measurements_per_chunk):
      # spread a single pass through the chromosome (a metaepoch) across multiple epochs, only loading dataset size into memory at each epoch
      self.n_ep_per_metaep = math.ceil(len(self.chunk_indexes) / self.n_epoch_chunks)
      self.subepoch_ind = self.n_ep_per_metaep - 1 # just so that on_epoch_end triggers a new metaepoch at start of training

  def on_epoch_end(self):
    """Updates indexes after each epoch and shuffles so that next metaepoch will not be done in same order
        i.e. the subepochs will not have the same sets of inds. Shuffling within subepoch needs to be handled on subclass"""
    self.subepoch_ind += 1
    if self.subepoch_ind >= self.n_ep_per_metaep:
      print('entering new metaepoch')
      if self.shuffle:
        print(' -- shuffling chunk indexes', flush=True)
        np.random.shuffle(self.chunk_indexes)
      self.subepoch_ind = 0

    print('subepoch {} of {}'.format(self.subepoch_ind+1, self.n_ep_per_metaep))

    self.epoch_chunk_indexes = self.chunk_indexes[self.subepoch_ind*self.n_epoch_chunks:(self.subepoch_ind+1)*self.n_epoch_chunks]
    if self.constant_size and len(self.epoch_chunk_indexes) < self.n_epoch_chunks:
      # TODO - check what this does exactly - it seems to just wrap around so that the number of epoch_chunk_indexes is the same for each epoch
      supp = self.n_epoch_chunks - len(self.epoch_chunk_indexes)
      self.epoch_chunk_indexes = np.concatenate([self.epoch_chunk_indexes, self.chunk_indexes[:supp]])
    print('Subsampled {} of {} chunks for this epoch'.format(len(self.epoch_chunk_indexes), len(self.chunk_indexes)))


class ChunkedTrainDataGeneratorHDF5(BaseChunkedDataGeneratorHDF5):

  """
  N.B. shuffling is handled entirely in on_epoch_end, which is also called on init to ensure data is shuffled before training
  """
      
  def __len__(self):
    if hasattr(self, 'epoch_chunk_indexes'):
      n_batches = math.ceil(len(self.epoch_chunk_indexes)*100 / self.batch_size)
    else:
      n_batches = math.ceil((self.total_chrom_chunks*100)/self.batch_size) # total n chunks is in units of h5py chunks
    self.n_batches = n_batches
    # we need to convert it into units of measurements
    print('{} size: {} batches of {} chunks covering {} chunks of {} bins each'.format(self.dataset, self.n_batches, self.batch_size,
                                                                                       self.batch_size*self.n_batches,1), flush=True)
    return self.n_batches

  def load_datasets(self):
    print('Loading data into memory', flush=True)
    chunk_list = []
    for ci in self.epoch_chunk_indexes:
      chrom, chunk_start = self.get_chunk_startbin_from_index(ci)
      chunk_list.append(chunk_start)

    all_train_x_chunks = []
    h5_filename = '{}_{}_targets.h5'.format(chrom, self.dataset)
    chrom_f = os.path.join(self.directory, '{}'.format(h5_filename))
    print('Loading {} data for chrom {} from file {}'.format(self.dataset, chrom, h5_filename))
    
    with h5py.File(chrom_f, 'r') as h5f:
      for chunk_start in chunk_list:
        chunk_targets = h5f['targets'][chunk_start:chunk_start+self.measurements_per_chunk]
        all_train_x_chunks.append(chunk_targets)

    self.train_x = np.concatenate(all_train_x_chunks)
    print('train data shape', self.train_x.shape)
    del all_train_x_chunks
    gc.collect()
 
  def release_memory(self):
    print('release memory ({})'.format(self.dataset), flush=True)
    if hasattr(self, 'train_x'):
      del self.train_x
    gc.collect()

  def on_epoch_end(self):
    self.release_memory()
    # load indexes of chunks for this metaepoch
    super().on_epoch_end()
    # load epoch data in memory from h5 based on indexes
    self.load_datasets()
    if self.shuffle:
      # n.b. this occurs at a different frequency (every epoch) than shuffling in parent class (every metaepoch)
      print('Shuffling data', flush=True)
      self.perm = np.random.permutation(self.train_x.shape[0])
      self.train_x = self.train_x[self.perm]

  def __getitem__(self, batch_index, debug=False):
    batch_start = batch_index*self.batch_size
    batch_stop = (batch_index+1)*self.batch_size
    if debug:
      print('Batch size', self.batch_size)

    batch_dict = defaultdict(list)

    # if batch_stop == batch_start:
    #   raise Exception('Empty batch')

    # if batch_stop - batch_start < batch_size:
    #   print(batch_stop, batch_start)

    # if debug:
    #   print(batch_start, batch_stop)
    #   print(targets.shape, y_obs.shape)

    y_obs = self.train_x[batch_start: batch_stop]
    expt_names = np.asarray(dataset_expts[self.dataset])
    batch_size = y_obs.shape[0]

    for i in range(batch_size):      
      drop_inds = np.random.choice(range(len(expt_names)), self.n_drop, replace=self.dataset not in ['train', 'all'])
      predict_inds = np.random.choice(drop_inds, self.n_predict, replace=False)
      predict_expts = expt_names[predict_inds]
      batch_dict['cell_ids'].append(np.asarray([train_cell2id[measurement[:3]] for measurement in predict_expts]))
      batch_dict['assay_ids'].append(np.asarray([train_assay2id[measurement[3:]] for measurement in predict_expts]))
      batch_dict['targets'].append(np.take(y_obs[i], predict_inds, axis=-1))
      batch_dict['drop_inds'].append(drop_inds)
      batch_dict['predict_inds'].append(predict_inds)

    batch_dict = {k: np.asarray(batch_vals) for k, batch_vals in batch_dict.items()}
    batch_y = batch_dict.pop('targets')
    batch_dict['y_obs'] = y_obs

    return batch_dict, batch_y

# TODO - implement
class ChunkedValDataGeneratorHDF5:
      
  def __len__(self):
    if hasattr(self, 'epoch_chunk_indexes'):
      n_batches = math.ceil(len(self.epoch_chunk_indexes)*100 / self.batch_size)
    else:
      n_batches = math.ceil((self.total_chrom_chunks*100)/self.batch_size) # total n chunks is in units of h5py chunks
    self.n_batches = n_batches
    # we need to convert it into units of measurements
    print('{} size: {} batches of {} chunks covering {} chunks of {} bins each'.format(self.dataset, self.n_batches, self.batch_size,
                                                                                       self.batch_size*self.n_batches,1), flush=True)
    return self.n_batches

  def load_datasets(self):
    print('Loading data into memory', flush=True)
    chunk_list = []
    for ci in self.epoch_chunk_indexes:
      chrom, chunk_start = self.get_chunk_startbin_from_index(ci)
      chunk_list.append(chunk_start)

    all_x_chunks = []
    all_train_x_chunks = []
    
    h5_filename = '{}_{}_targets.h5'.format(chrom, self.dataset)
    chrom_f = os.path.join(self.directory, '{}'.format(h5_filename))
    print('Loading {} data for chrom {} from file {}'.format(self.dataset, chrom, h5_filename))
    with h5py.File(chrom_f, 'r') as h5f:
      for chunk_start in chunk_list:
        chunk_targets = h5f['targets'][chunk_start:chunk_start+self.measurements_per_chunk]
        all_x_chunks.append(chunk_targets)

    self.x = np.concatenate(all_x_chunks)
    self.train_x = self.x
    
    print('TARGETS SHAPE', self.x.shape)
    print('INPUTS SHAPE', self.train_x.shape)
    del all_x_chunks
    del all_train_x_chunks
    gc.collect()

  def release_memory(self):
    print('release memory ({})'.format(self.dataset), flush=True)
    del self.x
    del self.train_x
    gc.collect()

  def on_epoch_end(self):
    self.release_memory()
    super().on_epoch_end()
    self.load_datasets()

    if self.shuffle:
      print('Shuffling data', flush=True)
      self.perm = np.random.permutation(self.x.shape[0])
      self.x = self.x[self.perm]
      self.train_x = self.x

  def __getitem__(self, batch_index, debug=False):
    # drop_inds = np.random.random_integers() - could get repeated things
    # print(batch_index)
    batch_start = batch_index*self.batch_size
    batch_stop = (batch_index+1)*self.batch_size
    if debug:
      print('Batch size', self.batch_size)
    # cell_ids = self.get_batch_cellids(batch_size)
    # batch_dict['assay_ids'] = self.get_batch_assayids(batch_size)
    batch_dict = defaultdict(list)

    # if self.dataset == 'train':
    #   y_obs = self.x[batch_start:batch_stop]
    #   targets = y_obs
    if batch_stop == batch_start:
      raise Exception('Empty batch')
    # else:
    if batch_stop - batch_start < batch_size:
      print(batch_stop, batch_start)

    y_obs = self.train_x[batch_start: batch_stop]
    targets = self.x[batch_start: batch_stop]

    if debug:
      print(batch_start, batch_stop)
      print(targets.shape, y_obs.shape)

    expt_names = np.asarray(dataset_expts[self.dataset])

    batch_size = y_obs.shape[0]
    for i in range(batch_size):
      try:
        a = targets[i]
      except:
        raise Exception('{} {} {} {}'.format(i, batch_size, str(targets.shape), str(y_obs.shape)))
      # print(i, targets[i])
      # if self.dataset == 'train':
      
      drop_inds = np.random.choice(range(len(expt_names)), self.n_drop, replace=self.dataset not in ['train', 'all'])
      if self.dataset in ['train','all'] and self.n_drop!=267: # 267 just for debugging purposes
        predict_inds = np.random.choice(drop_inds, self.n_predict, replace=False)
      else:
        predict_inds = np.arange(len(expt_names))
      predict_expts = expt_names[predict_inds]
      batch_dict['cell_ids'].append(np.asarray([train_cell2id[measurement[:3]] for measurement in predict_expts]))
      batch_dict['assay_ids'].append(np.asarray([train_assay2id[measurement[3:]] for measurement in predict_expts]))
      batch_dict['targets'].append(np.take(targets[i], predict_inds, axis=-1))
      batch_dict['drop_inds'].append(drop_inds)
      batch_dict['predict_inds'].append(predict_inds)

    batch_dict = {k: np.asarray(batch_vals) for k, batch_vals in batch_dict.items()}
    batch_y = batch_dict.pop('targets')
    batch_dict['y_obs'] = y_obs

    return batch_dict, batch_y
