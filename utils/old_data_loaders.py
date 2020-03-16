import math
import re
import gc
from itertools import islice
from collections import namedtuple, defaultdict

import h5py
import numpy as np
import pandas as pd

from keras.utils import Sequence

from utils.base_data_loaders import BaseDataGenerator
from utils.data_helpers import apply_transform
from utils.CONSTANTS import all_chromosomes, BINNED_CHRSZ, CHRSZ, train_cell2id,\
                            train_assay2id, data_dir, dataset_expts


class BaseDataGenerator(Sequence):
  def __init__(self, track_resolution=25, dataset='train',
               chroms='all', expt_names='all', context_size=1025,
               genome_file=None, measurements_per_chunk=1, chunks_per_batch=1,
               shuffle=True, transform=None, dataset_fraction=1,
               custom_splits=False, scaler=None, full_train=False,
               checkpoint_each_eval=False,
               debug=False, replace_gaps=False, subsample_each_epoch=False,
               use_metaepochs=False, assay_average=False, dataset_size=None, data_format=None,
               avg_types=[], chunk_axis=False, constant_size=True, **kwargs):
    """
     For chr21 validation I was using dataset fraction (a bit small). The corresponding size (in bins) would be
    """
    print('New data loader class')
    print('Transform:', transform)
    if chroms == 'all':
      chroms = all_chromosomes
    self.chroms = chroms
    self.dataset = dataset
    self.debug = debug
    self.scaler = scaler
    self.full_train = full_train
    self.chunk_axis = chunk_axis
    self.constant_size = constant_size
    self.checkpoint_each_eval = checkpoint_each_eval
    self.custom_splits = custom_splits
    print('using custom splits:', self.custom_splits)
    if not self.custom_splits:
      print('loading track ids')
      print('dataset', self.dataset, 'train_dataset', self.train_dataset)
      all_track_names = dataset_expts['train'] + dataset_expts['val']
      track2id = {t: i for i, t in enumerate(all_track_names)}
      if self.dataset == 'test':
        self.track_ids = [track2id[e] for e in dataset_expts['val']]
      else:
        self.track_ids = [track2id[e] for e in self.expt_names]

      if self.dataset == 'all' or self.train_dataset == 'all':
        self.train_track_ids = [track2id[e] for e in self.dataset_expts['train']+self.dataset_expts['val']]
      else:
        self.train_track_ids = [track2id[e] for e in self.dataset_expts['train']]
      print('n train track ids', len(self.train_track_ids))

    if not self.chunk_axis:
      assert measurements_per_chunk == 1
    self.transform = transform
    print('Multi obs', self.multi_obs)
    self.measurements_per_chunk = measurements_per_chunk # effectively number of bins per chunk
    self.chunks_per_batch = chunks_per_batch
    self.track_resolution = track_resolution
    self.context_size = context_size
    self.genome_file = genome_file
    self.shuffle = shuffle
    self.use_metaepochs = use_metaepochs
    self.dataset_size = dataset_size
    if use_metaepochs:
      subsample_each_epoch = True
    self.subsample_each_epoch = subsample_each_epoch
    self.assay_average = assay_average
    self.avg_types = avg_types
    if 'assay' in avg_types:
      self.assay_average = True
    self.data_format = data_format
    print('Assay av', self.assay_average)
    # self.shuffle_once = shuffle_once
    self.replace_gaps = replace_gaps

    assert not ((dataset_fraction<1) and dataset_size is not None), 'Cant set both dataset fraction and dataset size'
    self.use_reduced_dataset = (dataset_fraction < 1) or (dataset_size is not None)
    if dataset_fraction < 1 or dataset_size is None:
      # just take full chunks
      chunks_per_chrom = [math.floor(BINNED_CHRSZ[chrom]/self.measurements_per_chunk) for chrom in self.chroms]
    else:
      chunks_per_chrom = [math.ceil(BINNED_CHRSZ[chrom]/self.measurements_per_chunk) for chrom in self.chroms]
    self.cum_chunks_per_chrom = np.cumsum(chunks_per_chrom)
    print('chromosomes, cumulative chunks per chromosome: ', self.chroms, self.cum_chunks_per_chrom)
    print('Setting dataset fraction')
    self.set_dataset_fraction(dataset_fraction)
    print('Loading data')
    self.load_datasets()
    print('\n') # add some whitespace to stdout!

  def __len__(self):
    self.n_batches = math.ceil(self.total_n_chunks/self.chunks_per_batch)
    print('{} batches of {} chunks covering {} chunks of {} bins each'.format(self.n_batches, self.chunks_per_batch,
                                                                              self.total_n_chunks, self.measurements_per_chunk), flush=True)
    return self.n_batches

  def __getitem__(self, batch_index):
    """
     Generate one batch of data.
     A batch is of the form (X_seq, X_cell, X_assay), (y)
     Where X_seq is (batch_size x (measurements_per_chunk*25)+context_size (maybe plusorminus 1))
           X_cell is (batch_size x measurements_per_chunk)
           X_assay is (batch_size x measurements_per_chunk)
           y is (batch_size x measurements_per_chunk)
           (inputs to model need to be appropriately shaped)

    ## More complex methods will return additional outputs e.g. sequence; pos id
        These can define a __getitem__ method as:
         batch_cell, batch_assay, batch_posfeats, batch_values = super().__getitem__(batch_index)
         batch_sequences = self._get_sequence_item(batch_index)
         problem with this is that it requires get_chunk_data not to return sequence
         but that's fine - it's about separation. get_chunk_data handles the local observations
         separate methods handle other things.
         return batch_sequences, batch_cell, batch_assay, batch_feats, batch_values
    """
    # Generate indexes of the batch
    # print('Getting chunk indexes')
    # TODO could do this implicitly via generator comprehensions instead

    chunk_indexes = self.epoch_chunk_indexes[batch_index*self.chunks_per_batch:(batch_index+1)*self.chunks_per_batch]
    # batch_size = len(chunk_indexes)
    # tile_shape = (batch_size, 1)
    # if self.chunk_index:
    #   tile_shape = (batch_size, self.measurements_per_chunk, 1)
    # batch_cell_ids = np.tile(np.asarray([train_cell2id[measurement[:3]] for measurement in self.expt_names]),
    #                          tile_shape)
    # print(batch_cell_ids.shape)
    # batch_assay_ids = np.tile(np.asarray([train_cell2id[measurement[:3]] for measurement in self.expt_names]),
    #                           tile_shape)
    batch_cell_ids = []
    batch_assay_ids = []
    batch_values = []
    batch_posfeats = [] # averages
    for ci in chunk_indexes:
      chrom, chunk_start = self.get_chunk_startbin_from_index(ci)
      # maybe get_chunk_data should return a dict
      # an alternative would be
      # if self.chunk_axis: get chunk data
      # else: get bin data
      cell_ids, assay_ids, feats, values = self.get_chunk_data(chrom, chunk_start)

      batch_cell_ids.append(cell_ids)
      batch_assay_ids.append(assay_ids)
      batch_values.append(values)
      batch_posfeats.append(feats)

    batch_cell_ids = np.stack(batch_cell_ids) # batch_size, measurements_per_chunk, n_obs
    batch_assay_ids = np.stack(batch_assay_ids)
    batch_values = np.stack(batch_values) # batch_size, chunk_size, n_obs
    batch_posfeats = np.stack(batch_posfeats)
    # if self.dataset=='val':
    #     print(batch_cell_ids.shape, batch_values.shape)

    # print(batch_cell_ids.shape, batch_assay_ids.shape, batch_values.shape, batch_posfeats.shape)
    
    return [batch_cell_ids, batch_assay_ids, batch_posfeats], batch_values

  def on_epoch_end(self):
    """Updates indexes after each epoch"""
    print('on epoch end ({})'.format(self.dataset))
    # TODO could do this implicitly via generator comprehensions instead
    if self.use_metaepochs:
      self.subepoch_ind += 1

      if self.subepoch_ind >= self.n_ep_per_metaep:
        # print('resetting subepoch ind')
        print('entering new metaepoch')
        if self.shuffle:
          print(' -- shuffling chunk indexes', flush=True)
          np.random.shuffle(self.chunk_indexes)
        self.subepoch_ind = 0
        # print(self.)
      print('subepoch {} of {}'.format(self.subepoch_ind+1, self.n_ep_per_metaep))

    if self.shuffle == True and not self.use_metaepochs:
      np.random.shuffle(self.chunk_indexes)
    if self.use_reduced_dataset and self.subsample_each_epoch or self.use_metaepochs: # really self.use_metaepochs implies self.subsample_each_epoch
      # if self.subsample is true then chunk_indexes is longer than epoch_chunk_indexes
      self.epoch_chunk_indexes = self.chunk_indexes[self.subepoch_ind*self.total_n_chunks:(self.subepoch_ind+1)*self.total_n_chunks]
      if self.constant_size and len(self.epoch_chunk_indexes) < self.total_n_chunks:
        supp = self.total_n_chunks - len(self.epoch_chunk_indexes)
        self.epoch_chunk_indexes = np.concatenate([self.epoch_chunk_indexes, self.chunk_indexes[:supp]])
      print('Subsampled {} of {} chunks for this epoch'.format(len(self.epoch_chunk_indexes), len(self.chunk_indexes)))
    else:
      self.epoch_chunk_indexes = self.chunk_indexes
    # if self.debug:
    print(len(self.chunk_indexes))
    if not hasattr(self, 'subset'):
      print('Epoch chunks', self.epoch_chunk_indexes)

  def get_chunk_startbin_from_index(self, chunk_index):
    # if we have an explicit list of chunkstarts, look up; if not, compute using chrom lengths
    # TODO - check this works
    # print('figuring out start bin')
    # if not
    if hasattr(self, 'all_chunkstarts'):
      chrom, chunk_startbin = self.all_chunkstarts[chunk_index]
      chunk_startbin = int(chunk_startbin)
      return chrom, chunk_startbin
    
    for i, (chrom, cum_chunks) in enumerate(zip(self.chroms, self.cum_chunks_per_chrom)):
      # print(chrom)
      if chunk_index < cum_chunks:
        chrom_chunk_index = chunk_index - self.cum_chunks_per_chrom[i-1] if i > 0 else chunk_index
        chunk_startbin = self.measurements_per_chunk * chrom_chunk_index
        # print("here chunk_startbin", chunk_startbin)
        return chrom, chunk_startbin
    raise ValueError('Chunk index overflows data')

  def set_dataset_fraction(self, fraction=1):
    """
     The nice thing is that this can then be reset
     So we can use a single generator to train with 0.01 of the data 
     And then evaluate with the full dataset at the end of training
     Or after a certain number of batches.
    """
    # chunk indexes index collections of measurements_per_chunk bins - total number of these is recorded in cum_chunks_per_chrom
    # but the thing is that when we use region input we will effectively take all of the observations from a chunk...
    print('Replacing gaps:', self.replace_gaps)
    if self.replace_gaps:
      print('loading gap info')
      chunk_indexes = self.load_gap_info()
      self.chunk_indexes = np.asarray([ci for ci in chunk_indexes if ci < self.cum_chunks_per_chrom[-1]])
    else:
      self.chunk_indexes = np.arange(self.cum_chunks_per_chrom[-1])
    print(len(self.chunk_indexes), self.cum_chunks_per_chrom)
    if self.dataset_size is not None:
      self.total_n_chunks = min(self.dataset_size // self.measurements_per_chunk, len(self.chunk_indexes))
    else:
      self.total_n_chunks = math.floor(fraction * len(self.chunk_indexes))
    print('Total number of chunks to be used in {}: {} of {} possible ungapped chunks, of {} possible total chunks'.format(self.dataset, self.total_n_chunks,
          len(self.chunk_indexes), self.cum_chunks_per_chrom[-1]))
    if self.use_metaepochs:
      self.n_ep_per_metaep = math.ceil(len(self.chunk_indexes) / self.total_n_chunks)
      print('subepoch 1 of {}'.format(self.n_ep_per_metaep))
    
    self.subepoch_ind = 0

    if self.shuffle:
      print("Performing initial shuffle")
      np.random.shuffle(self.chunk_indexes)
    
    # BASICALLY UNLESS FRACTION < 1 AND SUBSAMPLE EACH EPOCH WE WANT epoch_chunk_indexes and chunk_indexes to be the same
    if self.use_reduced_dataset and self.subsample_each_epoch:
      self.epoch_chunk_indexes = self.chunk_indexes[:self.total_n_chunks]
      print('Subsampled {} of {} chunks for this epoch'.format(self.total_n_chunks, len(self.chunk_indexes)))
    elif self.use_reduced_dataset and not self.subsample_each_epoch:
      print('Subsampling {} of {} chunks for future usage'.format(self.total_n_chunks, len(self.chunk_indexes)))
      self.chunk_indexes = self.chunk_indexes[:self.total_n_chunks]
      self.epoch_chunk_indexes = self.chunk_indexes
      # convert from genomic coords (chunk_index) to arbitrary coords
      # this allows embedding matrix to be limited to size self.total_n_chunks * self.measurements_per_chuk
      self.cid2id = {c: i for i, c in enumerate(self.chunk_indexes)} # convert 
    else:
      # fraction == 1
      self.epoch_chunk_indexes = self.chunk_indexes

    assert self.total_n_chunks == len(self.epoch_chunk_indexes)
    if self.debug:
      print('Using chunk indexes:', self.epoch_chunk_indexes)
    self.dataset_fraction = fraction

  def load_gap_info(self):
    self.gaps = pd.read_csv('{}gap.txt'.format(data_dir), sep='\t',
                            names=['chromosome', 'start', 'end', 'chr_id',
                                   'gap_char','gap_len', 'gap_type', '-'])
    self.bins_with_gaps = defaultdict(list)
    chunk_indexes = []
    for i, chrom in enumerate(self.chroms):
      chrom_gaps = self.gaps[self.gaps['chromosome']==chrom]
      print(len(chrom_gaps))
      if i == 0:
        chrom_chunk_start = 0
      else:
        chrom_chunk_start = self.cum_chunks_per_chrom[i-1]

      end_chunk = chrom_chunk_start
      for ix, row in chrom_gaps.iterrows():

        start_bin = row['start'] // self.track_resolution
        start_chunk = chrom_chunk_start + (start_bin // self.measurements_per_chunk)
        # if start_chunk <= BINNED_CHRSZ[chrom]:
        # print(start_chunk, min(start_chunk, BINNED_CHRSZ[chrom]), end_chunk)
        # print(list(range(end_chunk, min(start_chunk, BINNED_CHRSZ[chrom]))))
        chunk_indexes += list(range(end_chunk, min(start_chunk, BINNED_CHRSZ[chrom]))) # I mean really this could just be range

        end_bin = row['end'] // self.track_resolution
        end_chunk = chrom_chunk_start + (end_bin // self.measurements_per_chunk)

        # TODO - maybe just save intervals. YES - just 
        # self.chunk_indexes += chunk_indexes[chunk_at_previous_start_bin: chunk_at_this_start_bin]
        if self.debug:
          print('Adding gaps', row['start'], row['end'], start_bin, end_bin, start_chunk, end_chunk)
        self.bins_with_gaps[chrom] += [b for b in range(start_bin, end_bin+1)]
    print('Chunks without gaps', len(chunk_indexes))
    return np.asarray(chunk_indexes)

class HDF5Generator:

  multi_obs = True
  def __init__(self, directory=None, expt_names=None, include_var=False,
               use_backup=False, checkpoint_each_eval=True, log_prefix='val_',
               preload_files=False, dataset='train', train_dataset=None, **kwargs):
    print('hdf5 gen init')
    if directory is None:
      directory = data_dir

    print('chckpoint', checkpoint_each_eval)
    if hasattr(self, 'dataset_expts'):
      # self.expt_names = self.dataset_expts[self.dataset]
      pass
    else:
      self.dataset_expts = dataset_expts
      self.expt_names = dataset_expts[dataset]
    self.train_dataset = train_dataset
    self.checkpoint_each_eval = checkpoint_each_eval
    self.log_prefix = log_prefix
    self.directory = directory
    self.preload_files = preload_files
    self.include_var = include_var
    self.use_backup = use_backup
    self.drop_experiments = False
    self.n_expts = len(self.expt_names)
    
    super().__init__(dataset=dataset, checkpoint_each_eval=checkpoint_each_eval, **kwargs)

  def load_datasets(self):
    self.data = {}
    self.train_data = {}

    self.avg_inds = [[] for a in self.avg_types]
    self.track_avg_inds = [] # list of indexes of tracks to be used to compute average for each track
    # self.track_avg_inds[0] - list of inds to compute avg features for 0th track (i.e. train_expt_names[0])
    self.n_avg_feats = len(self.avg_types)
    if self.include_var:
      self.n_avg_feats *= 2

    train_expt_names = self.dataset_expts['train']

    train_expts_by_type = defaultdict(list)
    for i, expt in enumerate(train_expt_names):
      cell_type = re.match('C\d{2}', expt).group(0)
      assay_type = re.search('M\d{2}', expt).group(0) 
      train_expts_by_type[cell_type].append(i)
      train_expts_by_type[assay_type].append(i)

    for i, expt in enumerate(self.expt_names):
      assay_type = re.search('M\d{2}', expt).group(0)
      cell_type = re.search('C\d{2}', expt).group(0)
      self.track_avg_inds.append([j for j in train_expts_by_type[assay_type] if j != i])
      for atype_ind, atype in enumerate(self.avg_types):
        assert atype in ['global', 'assay', 'cell']
        if atype == 'global':
          unself_inds = [j for j, e in enumerate(train_expt_names) if e != expt]
          assert len(unself_inds) == len(train_expt_names) - 1
          self.avg_inds[atype_ind].append(unself_inds)
        if atype == 'assay':
          self.avg_inds[atype_ind].append([j for j in train_expts_by_type[assay_type] if j != i])
        if atype == 'cell':
          self.avg_inds[atype_ind].append([j for j in train_expt_by_type[cell_type] if j != i])

    for chrom in self.chroms:
      if self.use_backup:
        print('USING BACKUP')
      self.data[chrom] = self.directory+'{}_{}_targets.h5{}'.format(chrom, self.dataset, '.backup' if self.use_backup else '')
      if self.assay_average or len(self.avg_inds)>0 or self.use_y:
        self.train_data[chrom] = self.directory+'{}_train_targets.h5{}'.format(chrom, '.backup' if self.use_backup else '')
    
    if self.genome_file is not None:
      # assert (self.context_size - self.track_resolution*self.) % 2 == 0
      # assert self.context_size % 25 == 0
      # assert (self.context_size / 25) % 2 == 1
      print('setting up genome')
      self.genome = pyfasta.Fasta(self.genome_file, key_fn=lambda key: key.split()[0])
      self.chrom2genomekey = {k.split()[0]: k for k in self.genome if k.split()[0] in self.chroms}
      print(self.chrom2genomekey)

  def get_chunk_data(self, chrom, chunk_start, *args):
    chunk_stop = min(chunk_start+self.measurements_per_chunk, BINNED_CHRSZ[chrom])
    targets = self.get_bin_values(chrom, chunk_start, chunk_start+1)
    cell_ids = np.asarray([train_cell2id[measurement[:3]] for measurement in self.expt_names])
    assay_ids = np.asarray([train_assay2id[measurement[3:]] for measurement in self.expt_names])
    # print("HDF5 get chunk data", cell_ids.shape, assay_ids.shape)
    if self.assay_average:
      if self.dataset=='train':
        assay_av = self.compute_avgbin_values(targets)
      elif self.dataset=='val':
        train_vals = self.get_bin_values(chrom, chunk_start, chunk_stop, dataset='train')
        assay_av = self.compute_avgbin_values(train_vals)
    if self.chunk_axis:
      cell_ids = np.tile(cell_ids, (chunk_stop-chunk_start, 1))
      assay_ids = np.tile(assay_ids, (chunk_stop-chunk_start, 1))
    else:
      targets = np.squeeze(targets)
    if self.assay_average:
      if self.chunk_axis:
        assay_av = np.squeeze(assay_av)
      return cell_ids, assay_ids, np.expand_dims(assay_av,-1), targets
    return cell_ids, assay_ids, targets

  def get_bin_values(self, chrom, chunk_start, chunk_stop, dataset=None):
    # print('get bin values', flush=True)
    # print(list(self.data.values())[0])
    if dataset is None:
      if self.preload_files:
        vals = self.data[chrom]['targets'][chunk_start: chunk_stop]
      else:
        with h5py.File(self.data[chrom], 'r') as h5f:
          vals = h5f['targets'][chunk_start: chunk_stop]
    elif dataset == 'train':
      # print('Train dataset bin values', self.train_data)
      if self.preload_files:
        vals = self.train_data[chrom]['targets'][chunk_start: chunk_stop]
      else:
        with h5py.File(self.train_data[chrom], 'r') as h5f:
          vals = h5f['targets'][chunk_start: chunk_stop]
    vals = apply_transform(vals, transform=self.transform)
    if self.drop_experiments:
      vals = vals[:,self.expt_inds]
    if self.chunk_axis:
      return vals
    return vals

  def cleanup(self):
    for f in self.data.values():
      f.close()

  def compute_avgbin_values(self, targets):
    # targets is shape (batch, measurements_per_sequence, n_obs)
    # TODO - calculate on fly
    batch_avgs = np.zeros((targets.shape[0], len(self.expt_names)))
    for i, avg_inds in enumerate(self.track_avg_inds):
      # print(avg_inds)
      if not avg_inds:
        continue # empty list
      avgs = np.mean(targets[:,avg_inds], axis=-1)
      batch_avgs[:,i] = avgs
    return batch_avgs

  def compute_multiavgbin_values(self, targets):
    # targets is shape (batch, measurements_per_sequence, n_obs)
    # TODO - calculate on fly
    all_avgs = []
    for atype, type_inds in zip(self.avg_types, self.avg_inds):
      type_avgs = np.zeros((targets.shape[0], len(self.expt_names), 1 if not self.include_var else 2))
      for i, inds in enumerate(type_inds):
        # print(avg_inds)
        if not inds:
          continue # empty list
        avgs = np.mean(targets[:,inds], axis=-1)
        type_avgs[:,i, 0] = avgs
        if self.include_var:
          type_avgs[:, i, 1] = np.std(targets[:,inds], axis=-1)
      all_avgs.append(type_avgs)
    all_avgs = np.concatenate(all_avgs, axis=-1)
    return all_avgs

class HDF5InMemDict(HDF5Generator, BaseDataGenerator):
  """
   Designed to be used without multiprocessing
   Costly step is load_datasets...but this only needs to run once per epoch

   For the HDF5 not in mem we also need to make use of our knowledge of chunks
   to reduce the overhead of file calls

   One way we could do this would be by structuring training not just by mini batch but by multi batch
     ->> either directly feed in 100 x chunks_per_batch x n_drop observations
     or load 100 x chunks_per_batch x n_drops observations from chunks_per_batch randomly chosen locations into memory
     then interleave the individual postions/measurements in separate batches

     And the obvious way to do this is to use this generator to load 1G~5G of data each 'epoch'
     shuffle the data
     and just do everything in memory!

     This should be blazing fast and remove the need for multiprocessing entirely.
     The only cost is at the data loading stage, but this happens only once per epoch and is
     100 times more efficient by leveraging HDF5's chunks
  """
  def __init__(self, *args, n_drop=50, n_predict=None, val_base_filename='{}_{}_targets.h5',
               train_dataset='train', dataset_name='targets', **kwargs):
    print('hdf5 in mem init')
    # dataset_fraction, load datasets will be called at this point
    # {}_selected_valbins.h5'
    self.n_drop = n_drop
    if n_predict is None:
      n_predict = n_drop
    self.n_predict = n_predict
    print('val base', val_base_filename)
    self.val_base_filename = val_base_filename
    self.dataset_name = dataset_name
    assert self.n_predict <= self.n_drop, 'n predict must be lt n drop'
    super().__init__(*args, measurements_per_chunk=100, preload_files=False,
                     chunk_axis=True, train_dataset=train_dataset, **kwargs) # chunking means this will load same amount of data 100x faster
    if self.shuffle:
      self.perm = np.random.permutation(self.x.shape[0])
      print('Shuffling targets')
      self.x = self.x[self.perm]
      if self.dataset not in ['train','all'] or self.train_x.shape != self.x.shape:
        print('Applying same permuation to y obs')
        self.train_x = self.train_x[self.perm]
      else:
        print('Settin y obs equal to targets')
        self.train_x = self.x
      
  def __len__(self):
    if hasattr(self, 'epoch_chunk_indexes'):
      n_batches = math.ceil(len(self.epoch_chunk_indexes)*100 / self.chunks_per_batch)
    else:
      n_batches = math.ceil((self.total_n_chunks*100)/self.chunks_per_batch) # total n chunks is in units of h5py chunks
    self.n_batches = n_batches
    # we need to convert it into units of measurements
    print('{} size: {} batches of {} chunks covering {} chunks of {} bins each'.format(self.dataset, self.n_batches, self.chunks_per_batch,
                                                                                       self.chunks_per_batch*self.n_batches,1), flush=True)
    return self.n_batches

  def load_datasets(self):
    print('Loading data into memory', flush=True)
    chunks_by_chrom = defaultdict(list)
    for ci in self.epoch_chunk_indexes:
      chrom, chunk_start = self.get_chunk_startbin_from_index(ci)
      chunks_by_chrom[chrom].append(chunk_start)

    if self.use_backup:
      print('USING BACKUP', flush=True)

    all_x_chunks = []
    all_train_x_chunks = []
    for chrom in self.chroms:
      # h5_filename = '{}_selected_valbins.h5'.format(chrom)
      # h5_filename = '{}_{}_targets.h5'.format(chrom, self.dataset if not self.custom_splits else 'all')
      if self.dataset == 'val':
        h5_filename = self.val_base_filename.format(chrom, self.dataset if not self.custom_splits else 'all')
      elif self.dataset == 'test':
        h5_filename = '{}_{}_targets.h5'.format(chrom, self.train_dataset)
      else:
        h5_filename = '{}_{}_targets.h5'.format(chrom, self.dataset if not self.custom_splits else 'all')
      chrom_f = self.directory+'{}{}'.format(h5_filename, '.backup' if self.use_backup else '')
      print('Loading {} data for chrom {} from file {}'.format(self.dataset, chrom, h5_filename))
      with h5py.File(chrom_f, 'r') as h5f:
        # why self.measurements_per_chunk=1 here? this is never the case...
        if self.dataset_fraction == 1 and self.dataset_size is None:
          print('Loading full target dataset')
          chunk_targets = h5f[self.dataset_name][:BINNED_CHRSZ[chrom]]
          # chunk_targets = h5f['targets'][:BINNED_CHRSZ[chrom]]
          if self.custom_splits or re.search('selected_valbins', self.val_base_filename):
            print('Also loading input data from {}'.format(h5_filename))
            chunk_yobs = chunk_targets[:, self.train_track_ids]
            chunk_targets = chunk_targets[:, self.track_ids]
            all_train_x_chunks.append(chunk_yobs)
          all_x_chunks.append(chunk_targets)

        else:
          for chunk_start in chunks_by_chrom[chrom]:
            # print('Loading target data')
            chunk_targets = h5f[self.dataset_name][chunk_start:chunk_start+self.measurements_per_chunk]
            if self.custom_splits or re.search('selected_valbins', self.val_base_filename):
              # print('Also loading input data from {}'.format(h5_filename))
              chunk_yobs = chunk_targets[:, self.train_track_ids]
              chunk_targets = chunk_targets[:, self.track_ids]
              all_train_x_chunks.append(chunk_yobs)
            all_x_chunks.append(chunk_targets)

      ### Needed only for loading validation data from a separate file
      if self.dataset not in ['train','all','test'] and not (re.search('selected_valbins', self.val_base_filename) or self.custom_splits):
        chrom_train_f = self.directory+'{}_train_targets.h5{}'.format(chrom, '.backup' if self.use_backup else '')
        print('Loading train (i.e. input) data for chrom {}'.format(chrom))

        with h5py.File(chrom_train_f, 'r') as h5f:
          if self.dataset_fraction == 1 and self.dataset_size is None:
            print('Loading full dataset')
            all_train_x_chunks.append(h5f['targets'][:BINNED_CHRSZ[chrom]])
          else:
            for chunk_start in chunks_by_chrom[chrom]:
              all_train_x_chunks.append(h5f['targets'][chunk_start:chunk_start+self.measurements_per_chunk])

    print('Applying transformation', self.transform)
    self.x = apply_transform(np.concatenate(all_x_chunks), transform=self.transform)
    if self.scaler is not None:
      if self.dataset in ['train','all'] and (not hasattr(self.scaler, 'scale_')):
        print('FITTING SCALER ON LOADED {} DATA'.format(self.dataset))
        self.scaler.fit(self.x, self.dataset_expts[self.dataset])
      print('Scaling targets')
      self.x = self.scaler.transform(self.x, self.dataset_expts[self.dataset])
    if all_train_x_chunks:
      self.train_x = apply_transform(np.concatenate(all_train_x_chunks), transform=self.transform)
      if self.scaler is not None:
        print('Scaling observations')
        self.train_x = self.scaler.transform(self.train_x, self.dataset_expts[self.dataset])
    else:
      self.train_x = self.x

    assert self.x.shape[0] == self.train_x.shape[0], 'shapes of inputs and targets {}, {} dont match'.format(self.train_x.shape[0],
                                                                                                             self.x.shape[0])
    
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
      if self.dataset not in ['train', 'all'] or self.train_x.shape != self.x.shape:
        self.train_x = self.train_x[self.perm]
      else:
        self.train_x = self.x

  def __getitem__(self, batch_index, debug=False):
    # drop_inds = np.random.random_integers() - could get repeated things
    # print(batch_index)
    batch_size = self.chunks_per_batch
    batch_start = batch_index*batch_size
    batch_stop = (batch_index+1)*batch_size
    if debug:
      print('Batch size', batch_size)
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

    expt_names = np.asarray(self.dataset_expts[self.dataset])

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