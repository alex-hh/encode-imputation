import sys, argparse, glob, os
import numpy as np
import pandas as pd
import pyBigWig

from utils.CONSTANTS import CHRSZ, train_expt_names, train_dir, data_dir
from utils.track_handlers import BigWig

# what we do is just extract tracks from bigwig and average them over 25bp windows
# memory requirement: 32 bytes per float. Roughly 32x2.5x10^8 = 10^10 bytes per chromosome ~ 1GB per chromosome
# the overall thing after averaging should be similar - you divide by 25 bp and multiply by 23 chromosomes

def main(bw, chrs, window_size=25, uncompressed=False):
    """
    Function taken from evaluation scripts - https://github.com/ENCODE-DCC/imputation_challenge/blob/master/build_npy_from_bigwig.py
    ToDo: refactor so that this belongs to BigWig class
    Returns:
        { chr: [] } where [] is a numpy 1-dim array
    """
    bw = BigWig(bw)
    bw_base = bw_file.split('.bigwig')[0]
    if uncompressed:
      current_chroms = [f.split('.')[1] for f in glob.glob(bw_base+'*.npz')]
    else:
      current_chroms = [f.split('.')[1] for f in glob.glob(bw_base+'*.gz.npz')]
    run_chroms = [c for c in chrs if c not in current_chroms]
    print('running chromosomes', run_chroms, flush=True)
    bw.bw_to_dict(run_chroms, window_size=window_size)
    for k, v in bw.binned_chroms.items():
      if uncompressed:
        np.savez(bw_base+'.'+k, v)
      else:
        np.savez_compressed(bw_base+'.'+k+'.gz', v) # used to be just savez

if __name__ == '__main__':
  # load filenames
  parser = argparse.ArgumentParser()
  parser.add_argument('bw_file')
  parser.add_argument('--chrom', nargs='+', # chr3 chr21
                      default=['all'])
  parser.add_argument('--uncompressed', action='store_true')
  args = parser.parse_args()
  if args.chrom == ['all']:
      args.chrom = ['chr' + str(i) for i in range(1, 23)] + ['chrX']
  main(bw_file, args.chrom, uncompressed=args.uncompressed)
