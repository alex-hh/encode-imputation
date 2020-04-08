import os, argparse
import h5py

from utils.CONSTANTS import all_chromosomes, n_expts, data_dir, BINNED_CHRSZ
from utils.track_handlers import TrackHandler


def main(dataset, chrom, directory=None):
  assert dataset in ['train', 'val']
  if directory is None:
    directory = data_dir
  
  chrom_len = BINNED_CHRSZ[chrom]
  n_obs = n_expts[dataset]

  print('Loading tracks', flush=True)
  th = TrackHandler(dataset=dataset, chroms=[chrom],
                    dataset_dir=os.path.join(directory, '{}/'.format('training' if dataset=='train' else 'validation/')))
  th.load_datasets()

  ## TODO - individual script for each expt?
  with h5py.File(os.path.join(directory,'{}_{}_targets.h5'.format(chrom, dataset)), 'w') as h5f:
    h5f.create_dataset('targets', shape=(chrom_len, n_obs), chunks=(100, n_obs), compression='lzf')
    for i, expt_name in enumerate(th.expt_names):
      print(expt_name, flush=True)
      track = th.data[expt_name][chrom]
      h5f['targets'][:,i] = track


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('dataset', help='Dataset (train/val)')
  parser.add_argument('chrom', help='Chromosome string e.g. chr21') # a string
  parser.add_argument('--directory', help='Base data directory')
  args = parser.parse_args()
  main(args.dataset, args.chrom, directory=args.directory)