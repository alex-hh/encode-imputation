import argparse

from utils.CONSTANTS import dataset_expts

from prediction.save_chrom_preds_checkpoint import main as save_model_chrom_preds
from prediction.save_average_prediction import main as save_ensembled_preds
from prediction.evaluate_preds import main as evaluate_preds
from prediction.save_bigwig import main as save_bigwig


pred_chroms = ['chr21'] # chromosomes to predict
# pred_chroms = ['chr{}'.format(i) for i in list(range(23))+['X']]

def main(model_name, expt_set, checkpoint_code, dataset='test'):
  for pred_chrom in pred_chroms:
    print('Saving preds for model {} on chromosome {}'.format(model_name, pred_chrom))
    save_model_chrom_preds(model_name, expt_set, pred_chrom, checkpoint_code, dataset=dataset)
    print('Evaluating preds')
    evaluate_preds(model_name, expt_set, pred_chrom, checkpoint_code)

  print('Gathering predictions into bigwig tracks')
  pred_track_list = dataset_expts[dataset]
  for track in pred_track_list:
    # gather predictions for all chromosomes for given track into a single bigwig file
    save_bigwig(model_name, expt_set, checkpoint_code, track, dataset=dataset)




if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('model_name')
  parser.add_argument('expt_set')
  parser.add_argument('checkpoint_code') # ep07.1
  # parser.add_argument('-model_list', nargs='+', required=True)
  # parser.add_argument('-chroms', nargs='+', required=True)
  parser.add_argument('--dataset', default='test')
  args = parser.parse_args()
  main(args.model_name, args.expt_set, args.checkpoint_code, dataset=args.dataset)