import argparse

from utils.CONSTANTS import dataset_expts, all_chromosomes

from prediction.save_chrom_preds_final_model import main as save_model_chrom_preds
from prediction.save_average_prediction import main as save_ensembled_preds
from prediction.save_bigwig import main as save_bigwig


# predictions will be generated for each of these models (identical models trained on the corresponding chrom)
imp_weights_list = ['chromschr1', 'chromschr3', 'chromschr4', 'chromschr5', 'chromschr6',
                    'chromschr7', 'chromschr8', 'chromschr9', 'chromschr10', 'chromschr11',
                    'chromschr12', 'chromschr14', 'chromschr15', 'chromschr16', 'chromschr17',
                    'chromschr18', 'chromschr19', 'chromschr20', 'chromschr21', 'chromschrX']

# pred_chroms = ['chr{}'.format(i) for i in list(range(23))+['X']]
def main(expt_set=None, checkpoint_code=None, dataset='test', pred_chroms=['chr21'],
         train_dataset='all', all_chroms=False, weights_list=None, data_directory=None,
         output_directory=None, moving_average=False):
  if weights_list is None or len(weights_list)==0:
    weights_list = imp_weights_list
  if all_chroms:
    pred_chroms = all_chromosomes
  
  for weights_name in weights_list:
    for pred_chrom in pred_chroms:
      print('Saving preds for model {} on chromosome {}'.format(weights_name, pred_chrom))
      save_model_chrom_preds(weights_name, pred_chrom, expt_set=expt_set, checkpoint_code=checkpoint_code,
                             dataset=dataset, train_dataset=train_dataset, moving_average=moving_average,
                             data_directory=data_directory, output_directory=output_directory)

  print('Ensembling')
  for pred_chrom in pred_chroms:
    # saves a separate npz array for each track and each chromosome
    ensemble_path = save_ensembled_preds(expt_set, pred_chrom, checkpoint_code,
                                         dataset=dataset, model_list=weights_list,
                                         directory=output_directory)

  if len(pred_chroms) == 23:
    print('Gathering predictions into bigwig tracks')
    pred_track_list = dataset_expts[dataset]
    for track in pred_track_list:
      # gather predictions for all chromosomes for given track into a single bigwig file
      save_bigwig(ensemble_path, expt_set, checkpoint_code, track, dataset=dataset, chroms=pred_chroms,
                  output_directory=output_directory)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--chroms', nargs='+', default=['chr21'], help="Chromosome list e.g. chr1 chr3; specify 'all' to include all chroms")
  parser.add_argument('--weights_names', nargs='+', help="Names of weight files to use to generate predictions")
  parser.add_argument('--checkpoint_code', default='14') # ep07.1
  # parser.add_argument('-model_list', nargs='+', required=True)
  parser.add_argument('--dataset', default='test')
  parser.add_argument('--expt_set', type=str, default=None)
  parser.add_argument('--train_dataset', default='all')
  parser.add_argument('--moving_average', action='store_true')
  parser.add_argument('--all_chroms', action='store_true')
  parser.add_argument('--data_directory', default=None)
  parser.add_argument('--output_directory', default=None)
  args = parser.parse_args()

  main(args.expt_set, args.checkpoint_code, dataset=args.dataset,
       pred_chroms=args.chroms, all_chroms=args.all_chroms,
       weights_list=args.weights_names, data_directory=args.data_directory,
       output_directory=args.output_directory, moving_average=args.moving_average,
       train_dataset=args.train_dataset)