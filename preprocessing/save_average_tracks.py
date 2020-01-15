import sys, argparse, glob, os
import numpy as np

from utils.CONSTANTS import data_dir, all_chromosomes, train_expt_names, val_expt_names
from utils.track_handlers import TrackHandler

# what we do is just extract tracks from bigwig and average them over 25bp windows
# memory requirement: 32 bytes per float. Roughly 32x2.5x10^8 = 10^10 bytes per chromosome ~ 1GB per chromosome
# the overall thing after averaging should be similar - you divide by 25 bp and multiply by 23 chromosomes

def main(expt_name):
  """
  Save per-assay averages based on training datasets
  We want to do this separately for each expt in the training set
  As well as for each experiment in the validation set (although these tracks have been provided already)
  """
  assert expt_name in train_expt_names + val_expt_names
  th = TrackHandler(dataset='train')
  folder_path = 'training' if expt_name in train_expt_names else 'validation'
  for chrom in all_chromosomes:
    avg = th.load_average(expt_name, chrom)
    np.savez_compressed(data_dir + 'baselines/average/{}/{}.{}.gz'.format(folder_path, expt_name, chrom), avg)

if __name__ == '__main__':
  # load filenames
  parser = argparse.ArgumentParser()
  parser.add_argument('expt_name')
  args = parser.parse_args()
  main(args.expt_name)
