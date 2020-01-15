import argparse
import glob

import h5py
import numpy as np

from .extract_numpy import main as save_to_npz
from .build_chrom_h5py import main as save_chrom_h5
from utils.CONSTANTS import all_chromosomes, data_dir


def main(dataset, directory=None, chroms=['chr21']):
  """
   Build hdf5 files of train, val and full (train+val) tracks for each chromosome.
   N.B. it is much faster if the individual steps e.g. save_to_npz are fully parallelised (over chroms and tracks)
   This is just intended as an illustration of the full pipeline, which could itself be run in parallel over chroms
  """
  if chroms == ['all']:
    chroms = all_chromosomes
  if directory is None:
    directory = data_dir
  # we're working at chromosome level so first extract chr level np from genome level bigwigs
  dataset_bigwigs = glob.glob(directory+'/*.bigwig')
  for bw in dataset_bigwigs:
    save_to_npz(bw, chroms)

  # for train, val gather chromosome np arrays into an h5 array ordered according to th.expt_names
  for chrom in chroms:
    print(chrom)
    for dataset in ['train', 'val']:
      save_chrom_h5(dataset, chrom)

    # combine train, val h5pys into single h5py for 'full' data ('all') - used as input to predict test
    with h5py.File(directory+'{}_train_targets.h5'.format(chrom), 'r') as h5f:
      train_vals = h5f['targets'][:]
    with h5py.File(directory+'{}_val_targets.h5'.format(chrom), 'r') as h5f:
      val_vals = h5f['targets'][:]

    all_vals = np.concatenate([train_vals, val_vals], axis=-1)
    print('Building full (train+val) dataset')
    with h5py.File(directory+'{}_all_targets.h5'.format(chrom), 'w') as h5f:
      h5f['targets'][:] = all_vals
    
  print('Done')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('dataset')
  parser.add_argument('-chrom_list', nargs='+', required=True, help="Chromosome list e.g. chr1 chr3; specify 'all' to include all chroms")
  parser.add_argument('--directory', default=None)
  args = parser.parse_args()
  main(args.dataset, chroms=args.chrom_list, directory=args.directory)