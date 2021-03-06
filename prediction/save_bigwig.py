import argparse, os
import pyBigWig
import numpy as np

from utils.CONSTANTS import output_dir, all_chromosomes, dataset_expts, BINNED_CHRSZ, CHRSZ


def main(model_imp_path, expt_set, checkpoint_code, val_track, dataset='test'):
  print('model name (ensemble code) {} expt set {} checkpoint {} track {} dataset {}'.format(model_imp_path, expt_set, checkpoint_code,
                                                                                             val_track, dataset))

  outdir = output_dir + '{}_bigwigs/{}/{}'.format(dataset, expt_set, model_imp_path)
  os.makedirs(outdir, exist_ok=True)

  bw = pyBigWig.open(outdir + "{}.bigwig".format(val_track), "w")
  bw.addHeader([(c, CHRSZ[c]) for c in all_chromosomes])
  
  for chrom in all_chromosomes:
    print(chrom)
    chrom_size = CHRSZ[chrom]
    
    starts = np.arange(0, chrom_size, 25, dtype=np.int64)
    ends = np.arange(25, chrom_size+25, 25, dtype=np.int64)
    chroms = np.array([chrom]*starts.shape[0])

    ends[-1] = chrom_size # convert from ends whose final entries will be (n-1, n) with n-1 < chrom_size and chrom_size < n < chrom_size+25
                          # to (n-1, chrom size) e.g. (chrom_size-3, chrom_size+22) would become (chrom_size-3, chrom_size). So we have a reduced size range as the final range.

    imp_file = output_dir + '{}_imputations/{}/{}/{}.{}.npz'.format(dataset, expt_set, model_imp_path, val_track, chrom)
    if os.path.exists(imp_file):
      print('Loading predicted values')
      values = np.load(imp_file)['arr_0']
      assert np.sum(values) != 0.0, 'VALUES IS EMPTY'
    else:
      bw.close()
      os.remove(outdir + "{}.bigwig".format(val_track))
      raise Exception('Couldnt find file in path {}'.format(imp_file))

    print(values.shape)
    assert values.shape[0] == starts.shape[0]
    bw.addEntries(chroms, starts, ends=ends, values=values)
  
  bw.close()
  print('Done')

  # load and check vals
  # bw = pyBigWig.open(outdir + "{}.bigwig".format(val_track))
  # print(bw.values('chr21', 0,250))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('model_imp_path')
  parser.add_argument('expt_set')
  parser.add_argument('checkpoint_code')
  parser.add_argument('val_track')
  parser.add_argument('--dataset', default='test')
  args = parser.parse_args()
  main(args.model_imp_path, args.expt_set, args.checkpoint_code, args.val_track, dataset=args.dataset)