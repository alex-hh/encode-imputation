import argparse, os
import pyBigWig
import numpy as np

from utils.CONSTANTS import output_dir, all_chromosomes, dataset_expts, BINNED_CHRSZ, CHRSZ


def main(model_imp_path, expt_set, checkpoint_code, val_track, dataset='test', chroms=None, output_directory=None):
  if chroms is None:
    chroms = all_chromosomes
  if output_directory is None:
    output_directory = output_dir
  print('model name (ensemble code) {} expt set {} checkpoint {} track {} dataset {} chroms {}'.format(model_imp_path, expt_set, checkpoint_code,
                                                                                                       val_track, dataset, ','.join(chroms)))

  outdir = os.path.join(output_directory, '{}_bigwigs'.format(dataset), '' if expt_set is None else expt_set, model_imp_path)
  os.makedirs(outdir, exist_ok=True)

  print(outdir)
  bw_dir = os.path.join(outdir, "{}.bigwig".format(val_track))
  print(bw_dir)
  bw = pyBigWig.open(os.path.join(outdir, "{}.bigwig".format(val_track)), "w")
  bw.addHeader([(c, CHRSZ[c]) for c in all_chromosomes])
  
  for chrom in chroms:
    print(chrom)
    chrom_size = CHRSZ[chrom]
    
    starts = np.arange(0, chrom_size, 25, dtype=np.int64)
    ends = np.arange(25, chrom_size+25, 25, dtype=np.int64)
    chroms = np.array([chrom]*starts.shape[0])

    ends[-1] = chrom_size # convert from ends whose final entries will be (n-1, n) with n-1 < chrom_size and chrom_size < n < chrom_size+25
                          # to (n-1, chrom size) e.g. (chrom_size-3, chrom_size+22) would become (chrom_size-3, chrom_size). So we have a reduced size range as the final range.
    # '/{}.{}.{}.npz'.format(t, chrom, checkpoint_code)
    checkpoint_str = '' if checkpoint_code is None else '.' + str(checkpoint_code)
    imp_file = os.path.join(output_directory, '{}_imputations'.format(dataset), '' if expt_set is None else expt_set,
                            model_imp_path, '{}.{}{}.npz'.format(val_track, chrom, checkpoint_str))
    if os.path.exists(imp_file):
      print('Loading predicted values')
      values = np.load(imp_file)['arr_0']
      assert np.sum(values) != 0.0, 'VALUES IS EMPTY'
    else:
      bw.close()
      os.remove(os.path.join(outdir, "{}.bigwig".format(val_track)))
      raise Exception('Couldnt find file for track {} chr {} in path {}'.format(val_track, chrom, imp_file))

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
  parser.add_argument('--chroms', nargs='+', default=['chr21'])
  parser.add_argument('--dataset', default='test')
  parser.add_argument('--output_directory', type=str, default=None)
  args = parser.parse_args()
  main(args.model_imp_path, args.expt_set, args.checkpoint_code, args.val_track, dataset=args.dataset, chroms=args.chroms,
       output_directory=args.output_directory)