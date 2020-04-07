import argparse

from utils.CONSTANTS import dataset_expts

from prediction.save_chrom_preds_final_model import main as save_model_chrom_preds
from prediction.evaluate_preds import main as evaluate_preds
from prediction.save_bigwig import main as save_bigwig


# pred_chroms = ['chr21'] # chromosomes to predict
# pred_chroms = ['chr{}'.format(i) for i in list(range(23))+['X']]

def main(model_name, expt_set, checkpoint_code, dataset='test', pred_chroms=['chr21']):
  for pred_chrom in pred_chroms:
    print('Saving preds for model {} on chromosome {}'.format(model_name, pred_chrom))
    save_model_chrom_preds(model_name, expt_set, pred_chrom, checkpoint_code, dataset=dataset)
    print('Evaluating preds')
    evaluate_preds(model_name, expt_set, pred_chrom, checkpoint_code)

  print('Gathering predictions into bigwig tracks')
  pred_track_list = dataset_expts[dataset]
  for track in pred_track_list:
    # gather predictions for all chromosomes for given track into a single bigwig file
    save_bigwig(model_name, expt_set, checkpoint_code, track, dataset=dataset, chroms=pred_chroms)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('model_name')
  parser.add_argument('checkpoint_code') # ep07.1
  # parser.add_argument('-model_list', nargs='+', required=True)
  parser.add_argument('-chroms', nargs='+', default=['chr21'])
  parser.add_argument('--dataset', default='test')
  parser.add_argument('--expt_set', type=str, default='chr21_reprod') # name of folder in which model weights are saved, and in which predictions are to be saved
  args = parser.parse_args()
  main(args.model_name, args.expt_set, args.checkpoint_code, dataset=args.dataset, pred_chroms=args.chroms)