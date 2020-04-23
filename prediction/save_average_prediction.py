import os, glob, argparse, json, re
from collections import defaultdict

import numpy as np
import pandas as pd

from utils.CONSTANTS import output_dir, all_chromosomes, BINNED_CHRSZ, dataset_expts


def extract_chrom_num(s):
  m = re.search('\d+', s)
  if m is None:
    return 24
  else:
    return int(m.group(0))

def main(expt_set, chrom, checkpoint_code, dataset='val', model_list=[], directory=None):
  if directory is None:
    directory = output_dir
  
  model_list = sorted(model_list, key=extract_chrom_num)
  print('Using the following {} models'.format(len(model_list)), model_list)
  ensemble_code = 'c' + 'c'.join([str(extract_chrom_num(e)) for e in model_list])
  ensemble_imp_path = 'averaged_preds-{}/{}'.format(checkpoint_code, ensemble_code)

  imp_dir = os.path.join(directory, '{}_imputations/{}/'.format(dataset, expt_set))
  os.makedirs(os.path.join(imp_dir, ensemble_imp_path), exist_ok=True)
  pred_track_names = dataset_expts[dataset]
    
  print('averaging chromosome: {}'.format(chrom))
  
  for t in pred_track_names:
    print(t, flush=True)
    model_count = 0
    avg = np.zeros(BINNED_CHRSZ[chrom])
    expts_included = []
    for m in model_list:
      imp_path = os.path.join(imp_dir, m, '{}.{}.{}.npz'.format(t, chrom, checkpoint_code))
      if os.path.exists(imp_path):
        vals = np.load(imp_path)['arr_0']
        assert vals.shape[0] == BINNED_CHRSZ[chrom], 'wrong shape: pred shape {} != chrom shape {}'.format(vals.shape[0],
                                                                                                           BINNED_CHRSZ[chrom])
        avg += vals
        model_count += 1
        expts_included.append(m)
      else:
        print('No imputations {} {} at path'.format(m, t), imp_path)

    avg /= model_count
    all_zeros = not np.any(avg)
    assert not all_zeros, 'EMPTY ARRAY, NOT SAVING'
    nans = np.isnan(avg).any()
    assert not nans, 'NANS in ARRAY, NOT SAVING'
    
    np.savez_compressed(os.path.join(imp_dir, ensemble_imp_path, '{}.{}.npz'.format(t, chrom)), avg)
    # save list of models which had predictions and were therefore included
    with open(os.path.join(imp_dir, ensemble_imp_path, '{}.{}_info.txt'.format(t, chrom)), 'w') as f:
      for expt in expts_included:
        f.write(expt+'\n')
  
  print('Done', flush=True)
  return ensemble_imp_path


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('expt_set')
  parser.add_argument('chrom')
  parser.add_argument('checkpoint_code') # 07.1
  parser.add_argument('dataset')
  parser.add_argument('--model_list', nargs='+', required=True, help='Model names e.g. chromschr21')
  parser.add_argument('--directory', default=None)
  args = parser.parse_args()
  main(args.expt_set, args.chrom, args.checkpoint_code, dataset=args.dataset,
       model_list=args.model_list, directory=args.directory)
