import math
import h5py
import pandas as pd
from keras.utils import Sequence

train_df = pd.read_csv(data_dir+'enhanced_metadata_training_data.tsv')
train_df.set_index('filename', inplace=True)
train_syn_ids = train_df['synapse_id']

train_expt_names = [f.split('.')[0] for f in train_df.index.values]
val_expt_names = [f.split('.')[0] for f in pd.read_csv(data_dir+'metadata_validation_data.tsv', sep='\t')['filename'].values]
dataset_expts = {'train': train_expt_names,
                 'val': val_expt_names,
                 'all': train_expt_names + val_expt_names}

def cell_assay_types_from_expts(expt_names):
  cell_types = sorted(list(set([re.match('C\d{2}', e).group(0) for e in expt_names])))
  assay_types = sorted(list(set([re.search('M\d{2}', e).group(0) for e in expt_names])))
  return cell_types, assay_types

dataset_celltypes = {}
dataset_assaytypes = {}
for k, expts in dataset_expts.items():
    cell_types, assay_types = cell_assay_types_from_expts(expts)
    dataset_celltypes[k] = cell_types
    dataset_assaytypes[k] = assay_types

train_cell2id = {c: i for i, c in enumerate(dataset_celltypes['train'])}
train_assay2id = {a: i for i, a in enumerate(dataset_assaytypes['train'])}


class TrainDataGen(Sequence):
  
  def __init__(self, h5py_file, train_inds, mode='train', shuffle=True, batch_size=256):
    with h5py.File(h5py_file, 'r') as h5f:
      targets = h5f['targets'][:]
    self.train_y = targets[:, train_inds]
    self.shuffle = shuffle
    ## if train inds == val inds what happens?
    self.batch_size = batch_size
    self.mode = mode
    if self.shuffle:
      print('Shuffling data')
      self.perm = np.random.permutation(self.train_x.shape[0])
      self.train_y = self.train_y[self.perm]
  
  def release_memory(self):
    print('release memory ({})'.format(self.dataset), flush=True)
    del self.train_y
    gc.collect()

  def __len__(self):
    return math.ceil(self.train_y.shape[0] / self.batch_size)

  def on_epoch_end(self):
    self.release_memory()

    if self.shuffle:
      print('Shuffling data')
      self.perm = np.random.permutation(self.train_x.shape[0])
      self.train_y = self.train_y[self.perm]

  def __getitem__(self, batch_index):
    batch_start = batch_index*self.batch_size
    batch_stop = (batch_index+1)*self.batch_size

    batch_dict = defaultdict(list)

    y_batch = self.train_y[batch_start: batch_stop]

    expt_names = np.asarray(self.dataset_expts[self.dataset])

    batch_size = y_obs.shape[0]
    for i in range(batch_size):
      
      drop_inds = np.random.choice(range(len(expt_names)), self.n_drop, replace=False)
      predict_inds = np.random.choice(drop_inds, self.n_predict, replace=False)

      predict_expts = expt_names[predict_inds]
      batch_dict['cell_ids'].append(np.asarray([train_cell2id[measurement[:3]] for measurement in predict_expts]))
      batch_dict['assay_ids'].append(np.asarray([train_assay2id[measurement[3:]] for measurement in predict_expts]))
      batch_dict['targets'].append(np.take(y_batch[i], predict_inds, axis=-1))
      batch_dict['drop_inds'].append(drop_inds)
      batch_dict['predict_inds'].append(predict_inds)

    batch_dict = {k: np.asarray(batch_vals) for k, batch_vals in batch_dict.items()}
    batch_y = batch_dict.pop('targets')
    batch_dict['y_obs'] = y_batch

    if self.mode == 'train':
      return batch_dict, batch_y
    else:
      return batch_dict

class ValDataGen(Sequence):
  def __init__(self, h5py_file, train_inds, val_inds, batch_size=256):
    with h5py.File(h5py_file, 'r') as h5f:
      targets = h5f['targets'][:]
    self.train_y = targets[:, train_inds]
    self.val_y = targets[:, val_inds]
    self.batch_size = batch_size
    self.val_expt_names = np.asarray(self.dataset_expts['val'])
  
  def release_memory(self):
    print('release memory ({})'.format(self.dataset), flush=True)
    del self.train_y
    gc.collect()

  def __len__(self):
    return math.ceil(self.train_y.shape[0] / self.batch_size)

  def __getitem__(self, batch_index):
    batch_start = batch_index*self.batch_size
    batch_stop = (batch_index+1)*self.batch_size

    batch_dict = defaultdict(list)

    y_batch = self.train_y[batch_start: batch_stop]
    val_y_batch = self.val_y[batch_start: batch_stop]

    batch_size = y_obs.shape[0]

    # these inds are actually all constant (except drop inds, which is ignored by the model at prediction time), 
    # so no need for the for loop really
    for i in range(batch_size):
      
      drop_inds = np.random.choice(range(self.train_y.shape[0])), self.n_drop, replace=False) # effectively just a placeholder input
      predict_inds = np.arange(len(self.val_expt_names))

      batch_dict['cell_ids'].append(np.asarray([train_cell2id[measurement[:3]] for measurement in self.val_expt_names]))
      batch_dict['assay_ids'].append(np.asarray([train_assay2id[measurement[3:]] for measurement in self.val_expt_names]))
      batch_dict['drop_inds'].append(drop_inds)
      batch_dict['predict_inds'].append(predict_inds)

    batch_dict = {k: np.asarray(batch_vals) for k, batch_vals in batch_dict.items()}
    batch_dict['y_obs'] = y_batch

    return batch_dict, val_y_batch
