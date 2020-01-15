import math
from collections import namedtuple, defaultdict

import numpy as np
import pandas as pd

from keras.utils import Sequence

from utils.track_handlers import TrackHandler
from utils.CONSTANTS import all_chromosomes, BINNED_CHRSZ, CHRSZ, train_cell2id,\
                            train_assay2id, data_dir, dataset_expts


if data_dir in ['data', 'data/']:
  baseline_dir = 'data/evaluation_data/'
else:
  baseline_dir = '/work/ahawkins/encodedata/baselines/'


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


class DictBaseGenerator(BaseDataGenerator):

  def __getitem__(self, batch_index):
    batch_dict = defaultdict(list)
    chunk_indexes = self.epoch_chunk_indexes[batch_index*self.chunks_per_batch:(batch_index+1)*self.chunks_per_batch]
    for ci in chunk_indexes:
      chrom, chunk_start = self.get_chunk_startbin_from_index(ci)
      chunk_dict = self.get_chunk_data(chrom, chunk_start)
      for k, chunk_vals in chunk_dict.items():
        batch_dict[k].append(chunk_vals)
    batch_dict = {k: np.asarray(batch_vals) for k, batch_vals in batch_dict.items()}
    batch_y = batch_dict.pop('targets')
    return batch_dict, batch_y

class NewBaseGenerator(BaseDataGenerator):

  def __getitem__(self, batch_index):

    chunk_indexes = self.epoch_chunk_indexes[batch_index*self.chunks_per_batch:(batch_index+1)*self.chunks_per_batch]
    observations, chunk_intervals = self.get_batch_observations(chunk_indexes)
    batch_size = observations.shape[0]
    batch_dict['cell_ids'] = self.get_batch_cellids(batch_size)
    batch_dict['assay_ids'] = self.get_batch_assayids(batch_size)
    features = self.get_batch_features(batch_intervals, cell_ids, assay_ids, observations) # a dict
    return [cell_ids, assay_ids, features], observations

  def get_batch_observations(self, chunk_indexes, measurements=None):
    batch_targets = []
    batch_intervals = []
    for ci in chunk_indexes:
      chrom, chunk_start = self.get_chunk_startbin_from_index(ci)
      targets = self.get_bin_values(chrom, chunk_start)
    return np.stack(batch_targets), batch_intervals


  def get_batch_features(self, batch_intervals, cell_ids, assay_ids, observations):
    all_avgs = []
    n_feats = len(self.avg_inds)
    if self.include_var:
      n_feats*=2
    for av_ind, (atype, type_inds) in enumerate(zip(self.avg_types, self.avg_inds)):
      type_avgs = np.zeros(observations.shape+(n_feats,))
      for i, inds in enumerate(type_inds):
        if self.chunk_axis:
          avgs = np.mean(targets[:,:,inds], axis=-1) # batch_size, measurements_per_chunk
          type_avgs[:,:,i,av_ind] = avgs
        else:
          avgs = np.mean(targets[:, inds], axis=-1) # batch_size
          type_avgs[:,i,av_ind] = avgs
      all_avgs.append(type_avgs)
    return all_avgs

    #   for i, inds in enumerate(type_inds):
    #     # print(avg_inds)
    #     if not inds:
    #       continue # empty list
    #     avgs = np.mean(targets[:,inds], axis=-1)
    #     type_avgs[:,i, 0] = avgs
    #     if self.include_var:
    #       type_avgs[:, i, 1] = np.std(targets[:,inds], axis=-1)
    #   all_avgs.append(type_avgs)
    # all_avgs = np.concatenate(all_avgs, axis=-1)
    # return all_avgs

  def get_batch_cellids(self, batch_size, measurements=None):
    tile_shape = (batch_size, 1)
    if self.chunk_axis:
      tile_shape = (batch_size, self.measurements_per_chunk, 1)
    return np.tile(np.asarray([train_cell2id[measurement[:3]] for measurement in self.expt_names]),
                              tile_shape)

  def get_batch_assayids(self, batch_size, measurements=None):
    tile_shape = (batch_size, 1)
    if self.chunk_axis:
      tile_shape = (batch_size, self.measurements_per_chunk, 1)
    return np.tile(np.asarray([train_assay2id[measurement[3:]] for measurement in self.expt_names]),
                              tile_shape)


class MemDataGenerator(BaseDataGenerator):
  multi_obs = False

  def __init__(self, use_compressed=True,**kwargs):
    """
      N.B. for a fully convolutional network we're effectively going to treat
           sequence_length as a receptive field size, and say we have 
           m overapping positions at which we have a rf of size sequence length,
           giving a total sequence length requirement
      Avocado generates data in this sequential way (taking thousands of neighbouring positions at a time)
      Here we generalise this idea via the notion of a chunk (a local stretch), and allowing for 
      multiple chunks per batch

      TODO - separate subsampling from shuffle
       subsample_each_epoch: if dataset_fraction < 1, select a new dataset_fraction set of examples each epoch
       shuffle: if true, shuffle chunk indexes at the start of each epoch
    """
    self.use_compressed = use_compressed
    super().__init__(**kwargs)

  def load_datasets(self):
    dataset_fpaths = {'train': 'training', 'val': 'validation'}
    print('setting up {} tracks'.format(self.dataset), flush=True)
    self.track_handler = TrackHandler(dataset=self.dataset, chroms=self.chroms,
                                      use_compressed=self.use_compressed,
                                      transform=self.transform)
    self.track_handler.load_datasets()
    if self.assay_average:
      avg_directory = baseline_dir+'average/{}/'.format(dataset_fpaths[self.dataset])
      print('Also loading average data from directory {}'.format(avg_directory), flush=True)
      self.avgtrack_handler = TrackHandler(dataset=self.dataset, chroms=self.chroms,
                                           use_compressed=self.use_compressed,
                                           transform=self.transform,
                                           dataset_dir=avg_directory
                                           )
      self.avgtrack_handler.load_datasets()
    else:
      print('Not loading average data', flush=True)
    self.n_expts = len(self.track_handler.expt_names)

    if self.genome_file is not None:
      # assert (self.context_size - self.track_resolution) % 2 == 0
      # assert self.context_size % 25 == 0
      # assert (self.context_size / 25) % 2 == 1
      print('setting up genome')
      self.genome = pyfasta.Fasta(self.genome_file, key_fn=lambda key: key.split()[0])
      self.chrom2genomekey = {k.split()[0]: k for k in self.genome if k.split()[0] in self.chroms}
      print(self.chrom2genomekey)

  def get_chunk_data(self, chrom, chunk_start):
    # TODO handle chunks that overflow - use some kind of masking / padding
    # --- be aware that as currently written this is going to break down when measurements per chunk > 1
    # TODO compute on the fly based on chromosome lengths
    # bins = list(np.arange(chunk_start, min(chunk_start+self.measurements_per_chunk, BINNED_CHRSZ[chrom])))
    
    chunk_stop = min(chunk_start+self.measurements_per_chunk, BINNED_CHRSZ[chrom])
    measurements = np.random.choice(self.track_handler.expt_names, chunk_stop-chunk_start)
    targets = self.get_bin_values(chrom, chunk_start, measurements) # (measurements_per_chunk, )
    cell_ids = [train_cell2id[measurement[:3]] for measurement in measurements]
    assay_ids = [train_assay2id[measurement[3:]] for measurement in measurements]
    
    # if self.chunk_axis:
    #   cell_ids = np.tile(cell_ids, (chunk_stop-chunk_start, 1))
    #   assay_ids = np.tile(assay_ids, (chunk_stop-chunk_start, 1))
    # else:
    #   targets = np.squeeze(targets)
    #   assay_av = np.squeeze(assay_av)
    # if self.assay_average:
    #   return cell_ids, assay_ids, np.expand_dims(assay_av,-1), targets
    # print(targets.shape, cell_ids.shape, assay_ids.shap)
    if self.assay_average:
      assay_av = self.get_avgbin_values(chrom, chunk_start, chunk_stop, measurements)
      # prin
      return cell_ids, assay_ids, np.expand_dims(assay_av, -1), targets
    return cell_ids, assay_ids, targets

  def get_bin_values(self, chrom, chunk_start, chunk_stop, measurements):
    # N.B. we need some way of ensuring the cell id
    return [self.track_handler.data[m][chrom][b] for m, b in zip(measurements, np.arange(chunk_start, chunk_stop))]

  def get_avgbin_values(self, chrom, chunk_start, chunk_stop, measurements):
    # N.B. we need some way of ensuring the cell id
    return [self.avgtrack_handler.data[m][chrom][b] for m, b in zip(measurements, np.arange(chunk_start, chunk_stop))]

