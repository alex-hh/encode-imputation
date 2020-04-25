import argparse
from prediction.save_bigwig import main as save_bigwig
from utils.CONSTANTS import dataset_expts


def main(model_name, expt_set=None, checkpoint_code=None, dataset='test', chroms=['chr21'], output_directory=None):
  pred_track_list = dataset_expts[dataset]
  print('Saving bigwigs for {} predictions on {} dataset (chroms {})'.format(model_name, dataset, ', '.join(chroms)))
  for track in pred_track_list:
    # gather predictions for all chromosomes for given track into a single bigwig file
    save_bigwig(model_name, expt_set, track, checkpoint_code=checkpoint_code, dataset=dataset, chroms=chroms,
                output_directory=output_directory)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('model_name')
  parser.add_argument('--expt_set', type=str, default=None)
  parser.add_argument('--checkpoint_code', type=str, default=None) # ep07.1
  parser.add_argument('--chroms', nargs='+', default=['chr21'])
  parser.add_argument('--dataset', default='test')
  parser.add_argument('--output_directory', type=str, default=None)
  args = parser.parse_args()
  main(args.model_name, expt_set=args.expt_set, checkpoint_code=args.checkpoint_code,
       dataset=args.dataset, chroms=args.chroms, output_directory=args.output_directory)