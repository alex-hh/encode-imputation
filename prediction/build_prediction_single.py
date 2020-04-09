import argparse

from utils.CONSTANTS import dataset_expts

from prediction.save_chrom_preds_final_model import main as save_model_chrom_preds
from prediction.evaluate_preds import main as evaluate_preds
from prediction.save_bigwig import main as save_bigwig


# pred_chroms = ['chr21'] # chromosomes to predict
# pred_chroms = ['chr{}'.format(i) for i in list(range(23))+['X']]

def main(model_name, expt_set=None, checkpoint_code=None, dataset='test', train_dataset='all',
         pred_chroms=['chr21'], moving_average=False, data_directory=None,
         output_directory=None):
  for pred_chrom in pred_chroms:
    print('Saving preds for model {} on chromosome {}'.format(model_name, pred_chrom))
    save_model_chrom_preds(model_name, pred_chrom, expt_set=expt_set, checkpoint_code=checkpoint_code,
                           dataset=dataset, train_dataset=train_dataset, moving_average=moving_average,
                           data_directory=data_directory, output_directory=output_directory)
    # print('Evaluating preds')
    # if dataset == 'val':
    # evaluate_preds(model_name, expt_set, pred_chrom, checkpoint_code, output_directory=output_directory)

  print('Gathering predictions into bigwig tracks')
  pred_track_list = dataset_expts[dataset]
  for track in pred_track_list:
    # gather predictions for all predicted chromosomes for given track into a single bigwig file
    save_bigwig(model_name, expt_set, checkpoint_code, track, dataset=dataset, chroms=pred_chroms,
                output_directory=output_directory)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('model_name')
  parser.add_argument('--checkpoint_code', default=None) # ep07.1
  # parser.add_argument('-model_list', nargs='+', required=True)
  parser.add_argument('--chroms', nargs='+', default=['chr21'])
  parser.add_argument('--dataset', default='test')
  parser.add_argument('--train_dataset', default='all')
  parser.add_argument('--expt_set', type=str, default=None) # name of folder in which model weights are saved, and in which predictions are to be saved
  parser.add_argument('--moving_average', action='store_true')
  parser.add_argument('--data_directory', type=str, default=None)
  parser.add_argument('--output_directory', type=str, default=None)
  args = parser.parse_args()
  main(args.model_name, expt_set=args.expt_set, checkpoint_code=args.checkpoint_code,
       dataset=args.dataset, train_dataset=args.train_dataset, pred_chroms=args.chroms,
       moving_average=args.moving_average, data_directory=args.data_directory,
       output_directory=args.output_directory)