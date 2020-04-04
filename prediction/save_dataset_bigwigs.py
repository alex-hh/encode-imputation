import argparse
from prediction.save_bigwig import main as save_bigwig
from utils.CONSTANTS import dataset_expts


def main(model_name, expt_set, checkpoint_code, dataset='test', chroms=['chr21']):
  pred_track_list = dataset_expts[dataset]
  print('Saving bigwigs for {} predictions on {} dataset (chroms {})'.format(model_name, dataset, ', '.join(chroms)))
  for track in pred_track_list:
    # gather predictions for all chromosomes for given track into a single bigwig file
    save_bigwig(model_name, expt_set, checkpoint_code, track, dataset=dataset, chroms=chroms)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('model_name')
  parser.add_argument('expt_set')
  parser.add_argument('checkpoint_code') # ep07.1
  parser.add_argument('-chroms', nargs='+', default=['chr21'])
  parser.add_argument('--dataset', default='test')
  args = parser.parse_args()
  main(args.model_name, args.expt_set, args.checkpoint_code, dataset=args.dataset, chroms=args.chroms)