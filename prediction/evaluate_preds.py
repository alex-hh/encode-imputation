import os, sys, argparse, json, re

from utils.track_handlers import Experiment, TrackHandler
from utils.CONSTANTS import dataset_expts, data_dir, output_dir


def main(model_name, expt_set, chrom, checkpoint_code, dataset='val', expt_suffix=None, gt_file_type='h5'):
  expt_dir = os.path.join(output_dir, '{}_imputations', '' if expt_set is None else expt_set, model_name)
  th = TrackHandler(dataset=dataset, dataset_dir=expt_dir, chroms=[chrom], transform=None,
                    expt_suffix='.{}.npz'.format(checkpoint_code) if expt_suffix is None else expt_suffix)

  global_dict, expt_dict = th.calc_mse([chrom], gt_file_type=gt_file_type)
  print('global mse for all {} {} expts:'.format(model_name, dataset), global_dict['global'][chrom], flush=True)
  print(global_dict)
  for e in expt_dict['global'].keys():
    print('{}:\t{}'.format(e, expt_dict['global'][e][chrom]))
  
  eval_metrics_dir = os.path.join(output_dir, 'eval_metrics/' + expt_set)
  os.makedirs(eval_metrics_dir, exist_ok=True)
  th.save_mses(os.path.join(eval_metrics_dir, '{}_{}_{}_{}_metrics.csv'.format(model_name, dataset, chrom, checkpoint_code)))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('model_name')
  parser.add_argument('expt_set')
  parser.add_argument('chrom')
  parser.add_argument('checkpoint_code')
  parser.add_argument('--dataset', default='val')
  parser.add_argument('--expt_suffix', default=None)
  parser.add_argument('--gt_file_type', default='h5')
  args = parser.parse_args()
  main(args.model_name, args.expt_set, args.chrom, args.checkpoint_code,
       dataset=args.dataset, expt_suffix=args.expt_suffix,
       gt_file_type=args.gt_file_type)