import math
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd

from keras.utils import Sequence

from utils.CONSTANTS import all_chromosomes, BINNED_CHRSZ, CHRSZ, train_cell2id,\
                            train_assay2id, data_dir, dataset_expts


class TrainDataGeneratorHDF5(Sequence):

  track_resolution = 25

  def __init__(self, batch_size=256, split_file=None, n_drop=50,
               directory=None, chrom='chr21', replace_gaps=False,
               dataset='train', shuffle=True):
    ## TODO add possibility to drop gaps
    self.config = locals()
    del self.config['self']
    if directory is None:
      directory = data_dir
    self.shuffle = shuffle
    self.batch_size = batch_size
    self.directory = directory
    self.chrom = chrom
    self.n_drop = n_drop

    train_h5_filename = '{}_{}_targets.h5'.format(chrom, dataset)
    train_chrom_f = self.directory + '{}'.format(train_h5_filename)
    print('Loading input data for chrom {} from file {}'.format(chrom, train_h5_filename))
    with h5py.File(train_chrom_f, 'r') as h5f:
      self.train_x = h5f['targets'][:BINNED_CHRSZ[chrom]]

    if replace_gaps:
      self.exclude_gaps()

    self.train_expt_names = np.asarray(dataset_expts[dataset])

    if self.shuffle:
      print('Shuffling data', flush=True)
      self.perm = np.random.permutation(self.train_x.shape[0])
      self.train_x = self.train_x[self.perm]

  def __len__(self):
    return math.ceil(self.train_x.shape[0]/self.batch_size)

  def exclude_gaps(self):
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
      end_bin = row['end'] // self.track_resolution
      self.bins_with_gaps += [b for b in range(start_bin, end_bin+1)]
    
    print('Excluding {} bins'.format(len(self.bins_with_gaps)))
    mask = np.ones(self.train_x.shape[0], dtype=bool) # all elements included/True.
    mask[self.bins_with_gaps] = False  # Set unwanted elements to False
    self.train_x = self.train_x[mask]
    # https://stackoverflow.com/questions/12518043/numpy-indexing-return-the-rest

  def on_epoch_end(self):
    if self.shuffle:
      print('Shuffling data', flush=True)
      self.perm = np.random.permutation(self.train_x.shape[0])
      self.train_x = self.train_x[self.perm]

  def __getitem__(self, batch_index, debug=False):
    batch_start = batch_index*self.batch_size
    batch_stop = (batch_index+1)*self.batch_size

    y_obs = self.train_x[batch_start: batch_stop]
    
    batch_dict = defaultdict(list)
    batch_size = y_obs.shape[0]
    for i in range(batch_size):      
      drop_inds = np.random.choice(self.train_x.shape[1], self.n_drop) # a placeholder
      predict_expts = self.train_expt_names[drop_inds]
      batch_dict['cell_ids'].append(np.asarray([train_cell2id[measurement[:3]] for measurement in predict_expts]))
      batch_dict['assay_ids'].append(np.asarray([train_assay2id[measurement[3:]] for measurement in predict_expts]))
      batch_dict['targets'].append(np.take(y_obs[i], drop_inds, axis=-1))
      batch_dict['drop_inds'].append(drop_inds)
      batch_dict['predict_inds'].append(drop_inds)

    batch_dict = {k: np.asarray(batch_vals) for k, batch_vals in batch_dict.items()}
    batch_y = batch_dict.pop('targets')
    batch_dict['y_obs'] = y_obs

    return batch_dict, batch_y

class ValDataGeneratorHDF5(Sequence):

  def __init__(self, batch_size=256, n_drop=50,
               directory=None, train_dataset='train',
               chrom='chr21'):
    self.config = locals()
    del self.config['self']
    if directory is None:
      directory = data_dir
    self.batch_size = batch_size
    self.directory = directory
    self.chrom = chrom
    self.n_drop = n_drop
    self.train_dataset = train_dataset

    h5_filename = '{}_{}_targets.h5'.format(chrom, self.train_dataset)
    chrom_f = self.directory+'{}'.format(h5_filename)
    print('Loading val input data for chrom {} from file {}'.format(chrom, h5_filename))
    with h5py.File(chrom_f, 'r') as h5f:
      self.train_x = h5f['targets'][:BINNED_CHRSZ[chrom]]
    
    val_h5_filename = '{}_val_targets.h5'.format(chrom)
    val_chrom_f = self.directory + '{}'.format(val_h5_filename)
    print('Loading val target data for chrom {} from file {}'.format(chrom, val_h5_filename))
    with h5py.File(val_chrom_f, 'r') as h5f:
      self.val_x = h5f['targets'][:BINNED_CHRSZ[chrom]]
      self.val_expt_names = dataset_expts['val']

    print('INPUTS SHAPE', self.train_x.shape)
    print('TARGETS SHAPE', self.val_x.shape)

  def __len__(self):
    return math.ceil(self.train_x.shape[0]/self.batch_size)

  def __getitem__(self, batch_index, debug=False):
    batch_start = batch_index*self.batch_size
    batch_stop = (batch_index+1)*self.batch_size

    y_obs = self.train_x[batch_start: batch_stop]
    targets = self.val_x[batch_start: batch_stop]
    
    batch_dict = defaultdict(list)
    batch_size = y_obs.shape[0]
    for i in range(batch_size):      
      drop_inds = np.random.choice(self.train_x.shape[1], self.n_drop) # a placeholder
      predict_inds = np.arange(len(self.val_expt_names))
      predict_expts = self.val_expt_names
      batch_dict['cell_ids'].append(np.asarray([train_cell2id[measurement[:3]] for measurement in predict_expts]))
      batch_dict['assay_ids'].append(np.asarray([train_assay2id[measurement[3:]] for measurement in predict_expts]))
      batch_dict['drop_inds'].append(drop_inds)
      batch_dict['predict_inds'].append(predict_inds)

    batch_dict = {k: np.asarray(batch_vals) for k, batch_vals in batch_dict.items()}
    batch_dict['y_obs'] = y_obs

    return batch_dict, targets

class TestDataGeneratorHDF5(Sequence):

  def __init__(self, train_dataset='train', batch_size=256, n_drop=50,
               directory=None, chrom='chr21'):
    self.config = locals()
    del self.config['self']
    if directory is None:
      directory = data_dir
    self.train_dataset = train_dataset
    self.batch_size = batch_size
    self.n_drop = n_drop
    self.directory = directory
    self.test_expt_names = np.asarray(dataset_expts['test'])
    self.chrom = chrom
    print('Test experiments:', self.test_expt_names)
    h5_filename = '{}_{}_targets.h5'.format(chrom, self.train_dataset)
    chrom_f = self.directory+'{}'.format(h5_filename)
    print('Loading test input data for chrom {} from file {}'.format(chrom, h5_filename))
    with h5py.File(chrom_f, 'r') as h5f:
      self.train_x = h5f['targets'][:BINNED_CHRSZ[chrom]]

  def __len__(self):
    return math.ceil(self.train_x.shape[0]/self.batch_size)

  def __getitem__(self, batch_index, debug=False):
    batch_start = batch_index*self.batch_size
    batch_stop = (batch_index+1)*self.batch_size

    y_obs = self.train_x[batch_start: batch_stop]

    batch_dict = defaultdict(list)
    batch_size = y_obs.shape[0]
    for i in range(batch_size):      
      drop_inds = np.random.choice(self.train_x.shape[1], self.n_drop) # a placeholder
      predict_inds = np.arange(len(self.test_expt_names))
      predict_expts = self.test_expt_names
      batch_dict['cell_ids'].append(np.asarray([train_cell2id[measurement[:3]] for measurement in predict_expts]))
      batch_dict['assay_ids'].append(np.asarray([train_assay2id[measurement[3:]] for measurement in predict_expts]))
      batch_dict['drop_inds'].append(drop_inds)
      batch_dict['predict_inds'].append(predict_inds)

    batch_dict = {k: np.asarray(batch_vals) for k, batch_vals in batch_dict.items()}
    batch_dict['y_obs'] = y_obs

    return batch_dict