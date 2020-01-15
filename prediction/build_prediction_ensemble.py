import argparse

from utils.CONSTANTS import dataset_expts

from prediction.save_chrom_preds_checkpoint import main as save_model_chrom_preds
from prediction.save_average_prediction import main as save_ensembled_preds
from prediction.save_bigwig import main as save_bigwig


# predictions will be generated for each of these models (identical models trained on the corresponding chrom)
model_list = ['chromschr1', 'chromschr3', 'chromschr4', 'chromschr5', 'chromschr6',
              'chromschr7', 'chromschr8', 'chromschr9', 'chromschr10', 'chromschr11',
              'chromschr12', 'chromschr14', 'chromschr15', 'chromschr16', 'chromschr17',
              'chromschr18', 'chromschr19', 'chromschr20', 'chromschr21', 'chromschrX']
# specify the chromosomes to predict
pred_chroms = ['chr21'] # chromosomes to predict
# pred_chroms = ['chr{}'.format(i) for i in list(range(23))+['X']]


def main(expt_set, checkpoint_code, dataset='test'):
  for model_chrom in model_chroms:
    model_name = 'chroms' + model_chrom # name of saved weights
    for pred_chrom in pred_chroms:
      print('Saving preds for model {} on chromosome {}'.format(model_name, pred_chrom))
      save_model_chrom_preds(model_name, expt_set, pred_chrom, checkpoint_code, dataset=dataset)

  print('Ensembling')
  for pred_chrom in pred_chroms:
    # saves a separate npz array for each track and each chromosomes
    ensemble_path = save_ensembled_preds(expt_set, pred_chrom, checkpoint_code, dataset=dataset, expt_list=model_list)

  print('Gathering predictions into bigwig tracks')
  pred_track_list = dataset_expts[dataset]
  for track in pred_track_list:
    # gather predictions for all chromosomes for given track into a single bigwig file
    save_bigwig(ensemble_path, expt_set, checkpoint_code, track, dataset=dataset)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('expt_set')
  parser.add_argument('checkpoint_code') # ep07.1
  # parser.add_argument('-model_list', nargs='+', required=True)
  parser.add_argument('--dataset', default='test')
  args = parser.parse_args()
  main(args.expt_set, args.checkpoint_code, dataset=args.dataset)