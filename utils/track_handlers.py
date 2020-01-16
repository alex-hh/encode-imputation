import glob, re
import itertools
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import pdb
import pyBigWig

from utils.data_helpers import apply_transform
from .CONSTANTS import CHRSZ, BINNED_CHRSZ, data_dir, train_expt_names, val_expt_names,\
                       all_chromosomes, dataset_expts, dataset_celltypes, dataset_assaytypes
from models.metrics import gwcorr, gwspear, evaluate_predictions

def cell_assay_types_from_expts(expt_names):
  cell_types = sorted(list(set([re.match('C\d{2}', e).group(0) for e in expt_names])))
  assay_types = sorted(list(set([re.search('M\d{2}', e).group(0) for e in expt_names])))
  return cell_types, assay_types

class BigWig:
  """
   Class to handle single bigwig files
   expt_name can be either just C20M15 or a full file path
  """

  def __init__(self, expt_name, data_folder='training_data/'):
    if len(expt_name.split('/')) > 1:
      # it's a full filepath
      # print(expt_name, flush=True)
      assert expt_name[-7:] == '.bigwig' or expt_name[-7:] == '.bigWig'
      self.track = expt_name
      self.base = expt_name[:-7]
      expt_name = self.base.split('/')[-1]
    else:
      self.base = data_dir + data_folder + expt_name
      self.track = data_dir + data_folder + expt_name + '.bigwig'
    self.binned_chroms = {}

  def load_chrnp(self, chromosome):
    if chromosome in self.binned_chroms:
      return self.binned_chroms[chromosome]
    return np.load(self.base + '.{}.npz'.format(chromosome))['arr_0']

  def slice(self, chromosome, start, end, binned=True):
    # memmap?
    if binned:
      return self.load_chrnp(chromosome)[start:end]
    else:
      return np.asarray(pyBigWig.open(self.track).values(chromosome, start, end))

  def np_chromosome_lengths(self, chromosomes='all'):
    if chromosomes == 'all':
      chromosomes = all_chromosomes
    
    lengths = [self.load_chrnp(chrom).shape[0] for chrom in chromosomes]
    return lengths

  def bw_to_dict(self, chrs, window_size=25):
    """
    Function taken from evaluation scripts - https://github.com/ENCODE-DCC/imputation_challenge/blob/master/build_npy_from_bigwig.py
    Each chromosome is binned into ((chrom_len-1)//window_size)+1 nonoverlapping bins of size window_size
    NaN values are converted into zeros before averaging over the bins
    Because the ends of the bigwig files contain NaNs - regions somehow not measured,
            a naive bin and then average is liable to cause problems in the first bin which contains nans.
            Perhaps the simplest solution would just be to use nanmean, and only replace nans after averaging.
            But I've stuck with the provided script for now
    """
    bw = pyBigWig.open(self.track)
    for c in chrs:
        print('Reading chromosome {} from bigwig...'.format(c), flush=True)
        chrom_len = bw.chroms()[c]
        # print(chrom_len, window_size)
        num_step = ((chrom_len-1)//window_size)+1 # number of bins ensuring all positions are included
        raw = bw.values(c, 0, chrom_len, numpy=True) # reshape raw vector as (num_step, window_size)
        raw.resize(num_step*window_size) # typically greater than chrom len
        # print number of nans (effectively 0s - we should ignore 0s somehow)
        # print(np.sum(np.isnan(raw)))
        raw = np.nan_to_num(raw) # pyBigWig returns nan for values out of bounds - convert to zero
        raw = np.reshape(raw, (-1, window_size)) # bin it
        result_per_chr = raw.mean(axis=1) # average over bins

        # special treatment for last step [i.e. last step with non nan values] (where the first nan is)
        # above averaging method does not work with the end step - because we've added zeros instead of nans
        # bw.intervals(c)[-1] is the last interval in bigwig
        # (248933861, 248934005, 0.08760000020265579)
        last_interval_end = bw.intervals(c)[-1][1] # find the end location of the last interval. after this we will have nans
        last_step = last_interval_end//window_size # where does our last valid window end
        start = last_step*window_size # where should our first special treatment window start
        end = min((last_step+1)*window_size, chrom_len)
        stat = bw.stats(c, start, end, exact=True)
        # pdb.set_trace()
        if stat[0] is None:
            result_per_chr[last_step]=0.0
        else:
            result_per_chr[last_step]=stat[0]

        self.binned_chroms[c] = np.array(result_per_chr)


class Experiment:
  def __init__(self, expt_name, data_folder, use_compressed=True,
               chroms='all', transform=None, expt_suffix=None):
    if expt_suffix is None:
      if use_compressed:
        expt_suffix = '.gz.npz'
      else:
        expt_suffix = '.npz'
    
    if chroms == 'all':
      chroms = all_chromosomes
    self.use_compressed = use_compressed
    self.transform = transform
    self.expt_name = expt_name
    self.expt_suffix = expt_suffix
    self.data_folder = data_folder
    self.chroms = chroms
    chrom_tracks = glob.glob(data_folder+expt_name+'*'+expt_suffix)
    chroms_with_tracks = [t.split('/')[-1].split('.')[1] for t in chrom_tracks]
    # print(chrom_tracks, data_folder+expt_name+'*'+expt_suffix)
    # print(chrom_tracks)
    for chrom in chroms:
      assert chrom in chroms_with_tracks, '{} missing for {} in {} with pattern {}'.format(chrom, expt_name, data_folder, expt_name+'*'+expt_suffix)

  def __str__(self):
    return self.expt_name

  # def __name__(self):
  #   return self.expt_name

  def load_chrnp(self, chromosome, interval_filter=None):
    np_file = self.data_folder + '{}.{}{}'.format(self.expt_name, chromosome, self.expt_suffix)
    vals = np.squeeze(np.load(np_file)['arr_0'])
    if interval_filter is not None:
      vals = interval_filter.filter_array(vals, chromosome, array_resolution=25)
    vals = apply_transform(vals, transform=self.transform)
    return vals

  def load_chroms(self):
    all_vals = []
    for chrom in self.chroms:
      all_vals.append(self.load_chrnp(chrom))
    return np.concatenate(all_vals)

  def max(self, chromosome):
    return np.max(self.load_chrnp(chromosome))

  @staticmethod
  def get_promoter_bins(chrom, gene_annotations, prom_loc=80):
    bins = []
    for line in gene_annotations:
      chrom_, start, end, _, _, strand = line.split()
      start = int(start) // window_size
      end = int(end) // window_size + 1

      if chrom_ != chrom:
        continue

      if strand == '+':
        bins += range(start-prom_loc, start)

      else:
        bins += range(end, end+prom_loc)
    return bins

  @staticmethod
  def get_enhancer_bins(chrom, enh_annotations):
    bins = []
    for line in enh_annotations:
      chrom_, start, end, _, _, _, _, _, _, _, _, _ = line.split()
      start = int(start) // window_size
      end = int(end) // window_size + 1

      if chrom_ != chrom:
        continue
      bins += range(start, end)
    return bins

  @staticmethod
  def get_gene_bins(chrom, gene_annotations):
    bins = []
    bins = []
    for line in gene_annotations:
      chrom_, start, end, _, _, strand = line.split()
      start = int(start) // window_size
      end = int(end) // window_size + 1

      if chrom_ != chrom:
        continue
      bins += range(start, end)
    return bins

  def avocado_correlation(self, chrom, dataset='test'):
    gt_expt_name = self.expt_name
    if dataset == 'val':
      dfolder = 'validation'
    elif dataset == 'test':
      dfolder = 'test'
    dataset_dir = '/work/ahawkins/encodedata/baselines/avocado/{}/'.format(dfolder)
    gt_expt = Experiment(gt_expt_name, dataset_dir, chroms=[chrom], transform=None)

    unfiltered_preds = self.load_chrnp(chrom, interval_filter=None)
    unfiltered_targets = gt_expt.load_chrnp(chrom, interval_filter=None)

    return gwcorr(unfiltered_targets, unfiltered_preds)

  def pairwise_correlation(self, chrom, comparison_expt):

    unfiltered_preds = self.load_chrnp(chrom, interval_filter=None)
    unfiltered_targets = comparison_expt.load_chrnp(chrom, interval_filter=None)

    return gwcorr(unfiltered_preds, unfiltered_targets)

  def bins_from_annotations(self, gene_annotations=None, enh_annotations=None, interval_filter=None):
    if gene_annotations is not None:
      gene_bins = self.get_gene_bins(chrom, gene_annotations)
      prom_bins = self.get_promoter_bins(chrom, gene_annotations)
    else:
      gene_bins, prom_bins = None, None

    if enh_annotations is not None:
      enh_bins = self.get_enhancer_bins(chrom, enh_annotations)
    else:
      enh_bins = None

    if interval_filter is not None:
      retain_bins = interval_filter.filter_bins(chrom, array_resolution=25)
    else:
      retain_bins = None
    return gene_bins, prom_bins, enh_bins, retain_bins

  def chrom_squared_errors_h5(self, chrom, dataset, gt_expt_name=None):
    if gt_expt_name is None:
      gt_expt_name = self.expt_name
    error_msg = "Ground truth experiment cannot be inferred from experiment name"
    error_msg += "so must be supplied, or if supplied must be a valid experiment name\n"
    error_msg += "gt_expt_name: \t{}".format(gt_expt_name)
    assert gt_expt_name in dataset_expts[dataset], error_msg
    gt_expt_ind = dataset_expts[dataset].index(gt_expt_name)

    # TODO put this into track from h5 helper function
    h5_filename = '{}_{}_targets.h5'.format(chrom, dataset)
    chrom_f = data_dir + '{}'.format(h5_filename)
    
    with h5py.File(chrom_f, 'r') as h5f:
      unfiltered_targets = h5f['targets'][:BINNED_CHRSZ[chrom], gt_expt_ind]

    unfiltered_preds = self.load_chrnp(chrom, interval_filter=None)
    assert unfiltered_preds.shape == unfiltered_targets.shape, 'prediction shape {} does not match target shape {}'.format(preds.shape, targets.shape)
    return unfiltered_preds, unfiltered_targets

  def chrom_squared_errors_npz(self, chrom, gt_expt_name=None):
    from utils.CONSTANTS import train_dir, val_dir
    if gt_expt_name is None:
      gt_expt_name = self.expt_name
    error_msg = "Ground truth experiment cannot be inferred from experiment name"
    error_msg += "so must be supplied, or if supplied must be a valid experiment name\n"
    error_msg += "gt_expt_name: \t{}".format(gt_expt_name)
    assert gt_expt_name in train_expt_names + val_expt_names, error_msg
    dataset_dir = train_dir if gt_expt_name in train_expt_names else val_dir
    gt_expt = Experiment(gt_expt_name, dataset_dir, chroms=[chrom], transform=self.transform)

    # maybe to handle filters we should set filtered segments to nan
    # or just separately load filtered preds and unfiltered preds
    unfiltered_preds = self.load_chrnp(chrom, interval_filter=None)
    unfiltered_targets = gt_expt.load_chrnp(chrom, interval_filter=None)
    assert unfiltered_preds.shape == unfiltered_targets.shape, 'prediction shape {} does not match target shape {}'.format(preds.shape, targets.shape)
    return unfiltered_preds, unfiltered_targets

  def chrom_squared_errors(self, chrom, dataset, gt_file_type='h5', gt_expt_name=None, interval_filter=None,
                           gene_annotations=None, enh_annotations=None):
    if gt_file_type == 'h5':
      unfiltered_preds, unfiltered_targets = self.chrom_squared_errors_h5(chrom, dataset, gt_expt_name=gt_expt_name)
    elif gt_file_type == 'npz':
      unfiltered_preds, unfiltered_targets = self.chrom_squared_errors_npz(chrom, gt_expt_name=gt_expt_name)
    else:
      raise ValueError('file type must be either h5 or npz')
    gene_bins, prom_bins, enh_bins, retain_bins = self.bins_from_annotations(gene_annotations=gene_annotations,
                                                                             enh_annotations=enh_annotations,
                                                                             interval_filter=interval_filter)

    return evaluate_predictions(unfiltered_preds, unfiltered_targets, gene_bins=gene_bins,
                                promoter_bins=prom_bins, enhancer_bins=enh_bins, retain_bins=retain_bins)

  def chrom_mse(self, chrom, interval_filter=None):
    sse_dict, count_dict = self.chrom_squared_errors(chrom, interval_filter=interval_filter)
    return {k: sse_dict[k] / count_dict[k] for k in sse_dict.keys()}
    # return np.mean(self.chrom_squared_errors(chrom, interval_filter=interval_filter))

class TrackHandler:
  def __init__(self, dataset='train', chroms='all',
               expt_names='all', use_compressed=True,
               transform=None, dataset_dir=None, expt_suffix=None,
               split_file=None):
    if chroms == 'all':
      chroms = all_chromosomes

    if split_file is not None:
      split_df = pd.read_csv(split_file)
      val_track_names = split_df[split_df['set']=='V']['track_code'].values
      train_track_names = split_df[split_df['set']=='T']['track_code'].values
      if dataset == 'val':
        expt_names = val_track_names
      elif dataset == 'train':
        expt_names = train_track_names
      cell_types, assay_types = cell_assay_types_from_expts(expt_names)

    else:
      cell_types = dataset_celltypes[dataset]
      assay_types = dataset_assaytypes[dataset]
      if expt_names == 'all':
        expt_names = dataset_expts[dataset]
    
    self.use_compressed = use_compressed
    self.transform = transform
    self.chroms = chroms
    self.expt_names = expt_names
    self.dataset = dataset
    self.expts = []
    self.expts_by_type = defaultdict(list)
    self.dataset_dir = dataset_dir
    self.cell_types = cell_types
    self.assay_types = assay_types
    
    for expt_name in self.expt_names:
      cell_type = re.match('C\d{2}', expt_name).group(0) # overkill - could just slice
      assay_type = re.search('M\d{2}', expt_name).group(0)
      expt = Experiment(expt_name, self.dataset_dir,
                        chroms=chroms,
                        use_compressed=use_compressed,
                        transform=transform,
                        expt_suffix=expt_suffix)
      self.expts_by_type[cell_type].append(expt)
      self.expts_by_type[assay_type].append(expt)
      self.expts.append(expt)
      assert cell_type in self.cell_types and assay_type in self.assay_types

  def load_average(self, expt_name, chrom):
    avg = np.zeros(BINNED_CHRSZ[chrom])
    assay_type = re.search('M\d{2}', expt_name).group(0)
    count = 0
    avgd_expts = []
    for expt in self.expts_by_type[assay_type]:
      if expt.expt_name != expt_name:
        avg += expt.load_chrnp(chrom)
        count += 1
        avgd_expts.append(expt.expt_name)
    if count > 0:
      avg /= count
    print('Averaged over {} expts: '.format(count), avgd_expts)
    return avg

  def pairwise_corrs(self, chrom):
    expt_pairs = itertools.combinations(self.expts, 2)
    expt_pairs = [p for p in expt_pairs]
    sel_pairs = np.random.choice(np.arange(len(expt_pairs)), 50)
    for pair_ind in sel_pairs:
      pair = expt_pairs[pair_ind]
      corr = pair[0].pairwise_correlation(chrom, pair[1])
      print(corr)

  def load_datasets(self):
    """
     Load all datasets into memory (e.g. to train a model)
    """
    self.data = defaultdict(dict)
    for expt in self.expts:
      for chrom in self.chroms:
       self.data[expt.expt_name][chrom] = expt.load_chrnp(chrom)

  def type_average(self, type_code, base_outfile=None, write_out=False):
    """
    type_code can be either a cell type code OR an assay type code
    remember that whereas bigwig files are at 1bp resolution, np are at 25bp
    """
    if base_outfile is None:
      base_outfile = '{}statistics/{}_average'.format(data_dir, type_code) + '.{}.npz'
    result_dict = {}
    type_expts = self.expts_by_type[type_code]
    for chrom in self.chroms:
      av_vals = np.zeros(BINNED_CHRSZ[chrom])
      for expt in type_expts:
        av_vals += expt.load_chrnp(chrom)
      av_vals/=len(type_expts)
      if write_out:
        np.savez_compressed(base_outfile.format(chrom), av_vals)
      else:
        result_dict[chrom] = av_vals
    if not write_out:
      return result_dict

  def test_average(self, assay_code):
    assay_averages = glob.glob('/work/ahawkins/encodedata/baselines/average/*{}*.bigwig')
    gs = assay_averages[0]
    import pyBigWig
    gstrack = pyBigWig.open(gs)
    for start, end, val in gstrack.intervals():
      assert npvals[start//25] == val

  def assay_average(assay_code, chromosomes=['all'],
                    base_outfile=None, write_out=False):
    return self.type_average(assay_code, base_outfile=None, write_out=False)

  def cell_average(cell_code, chromosomes=['all'],
                    base_outfile=None, write_out=False):
    return self.type_average(cell_code, base_outfile=None, write_out=False)

  def slice(self, chrom, start, end):
    Y = np.zeros((end-start, len(self.expts))) # num datapoints x num features
    for i, e in enumerate(self.expts):
      # print('Loading expt', e)
      Y[:, i] = e.load_chrnp(chrom)[start:end]
    return Y

  def filter_constant(self, chrom, window_size=400, thresh=1e-5):
    bins = {}
    n_windows = BINNED_CHRSZ[chrom]//window_size
    for i, e in enumerate(self.expts):
      y = e.load_chrnp(chrom)[:n_windows*window_size]
      y = y.reshape((n_windows, window_size))
      y_stds = np.std(y, axis=1)
      # print(np.sum(np.argwhere(y_stds<thresh)))
      bins = set(list(np.argwhere(y_stds >= thresh).reshape(-1)))
      bins = bins.union(bins)
    self.nonconstant_bins = bins
    return sorted(list(bins))

  def avo_correlation(self, chrom):
    expt_corrs = {}
    for i, expt in enumerate(self.expts):
      corr = expt.avocado_correlation(chrom, dataset=self.dataset)
      print(expt.expt_name, corr)
      expt_corrs[expt] = corr
    return expt_corrs

  def calc_mse(self, chroms=['chr21'], gt_file_type='h5', interval_filter=None, gene_annotations=None, enh_annotations=None):
    """
    sse: sum squared error
    """
    self.global_mse = defaultdict(dict)
    self.expt_mses = defaultdict(lambda: defaultdict(lambda: defaultdict(int))) # mses by subset by expt by chromosome
    self.expt_sses = defaultdict(lambda: defaultdict(lambda: defaultdict(int))) # sum squared error by subset by expt by chromosome
    self.expt_obs = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    total_error_dict = defaultdict(int)
    total_obs_dict = defaultdict(int)
    # total_error = 0
    # total_obs = 0
    assert len(chroms) == 1, 'gwspear, gwcorr assume only a single eval chrom'
    # total_error1pc = 0
    # total_obs1pc = 0
    # chroms_str = '-'
    for chrom in chroms:
      for i, expt in enumerate(self.expts):
        if i % 100 == 0:
          print(i)
        # errors = expt.chrom_squared_errors(chrom, interval_filter=interval_filter)
        # total_error += np.sum(errors) # TODO make np.nansum(errors)
        # total_obs += errors.size
        # total_error += np.nansum(errors)
        # total_obs += np.count_nonzero(~np.isnan(errors))

        sse_dict, count_dict  = expt.chrom_squared_errors(chrom, self.dataset, gt_file_type=gt_file_type,
                                                          interval_filter=interval_filter,
                                                          gene_annotations=gene_annotations,
                                                          enh_annotations=enh_annotations)
        for k, sse in sse_dict.items():
          # error_keys = ['global_mse', 'top1_mse']
          # for k in error_keys
          # errors = error_dict[k]
          # self.expt_mses[k][expt.expt_name][chrom] = np.mean(errors)
          # self.expt_sses[k][expt.expt_name][chrom] = np.sum(errors)
          # total_error_dict[k] += np.sum(errors)
          total_error_dict[k] += sse
          total_obs_dict[k] += count_dict[k]
          assert count_dict[k] > 0, 'Count of {} for metric {} for expt {} invalid'.format(count_dict[k], k, expt.expt_name)
          self.expt_mses[k][expt.expt_name][chrom] = sse / count_dict[k]
          self.expt_sses[k][expt.expt_name][chrom] = sse
          self.expt_obs[k][expt.expt_name][chrom] = count_dict[k]
          self.expt_sses[k][expt.expt_name]['-'.join(chroms)] += sse
          self.expt_obs[k][expt.expt_name]['-'.join(chroms)] += count_dict[k]
        
        # self.expt_mses[expt.expt_name][chrom] = np.mean(errors)
        # self.expt_sses[expt.expt_name][chrom] = np.sum(errors)
        # self.expt_obs[expt.expt_name][chrom] = errors.size
        # self.expt_sses[expt.expt_name]['-'.join(chroms)] += np.sum(errors)
        # self.expt_obs[expt.expt_name]['-'.join(chroms)] += errors.size
    
    # # global calculations
    # total_error /= total_obs
    # # total_error1pc /= total_obs1pc
    # self.global_mse['-'.join(chroms)] = total_error
    # self.global_mse['filter'] = str(interval_filter)
    # # self.global_mse['-'.join(chroms)+'_top1'] = total_error1pc

    for metric, total_error in total_error_dict.items():
      total_obs = total_obs_dict[metric]
      total_error /= total_obs
      self.global_mse[metric]['-'.join(chroms)] = total_error
    self.global_mse['filter'] = str(interval_filter)

    for metric, metric_dict in self.expt_mses.items():
      for expt in self.expts:
        metric_dict[expt.expt_name]['-'.join(chroms)] = self.expt_sses[metric][expt.expt_name]['-'.join(chroms)]
        metric_dict[expt.expt_name]['-'.join(chroms)] /= self.expt_obs[metric][expt.expt_name]['-'.join(chroms)]

    return self.global_mse, self.expt_mses

  def save_mses(self, output_path=None):
    # | chromosomes | expt_name | filter_type | transform | mse
    if output_path is None:
      output_path = self.dataset_dir + 'mses.csv'

    with open(output_path, 'a') as outf:
      filter_name = self.global_mse.pop('filter')
      for metric, metric_dict in self.global_mse.items():
        for chrom_list, global_mse in metric_dict.items():
          outf.write('{},{},{},{},{},{}\n'.format(chrom_list, metric, 'global', filter_name, self.transform, global_mse))
        for expt_name in self.expt_mses[metric].keys():
          mse = self.expt_mses[metric][expt_name][chrom_list]
          outf.write('{},{},{},{},{},{}\n'.format(chrom_list, metric, expt_name, filter_name, self.transform, mse))

  def chrom_mse(self, chrom, interval_filter=None):
    self.mse_by_expt = {} # c.f. np prediction evaluator
    squared_error = 0
    n_vals = 0
    for expt in self.expts:
      error_dict = expt.chrom_squared_errors(chrom, interval_filter=interval_filter)
      squared_error += np.sum(error_dict['global_mse'])
      print('mse shape, mse size', error_dict['global_mse'].shape, error_dict['global_mse'].size)
      n_vals += error_dict['global_mse'].size
    print('Total squared error {} total n vals {}'.format(squared_error, n_vals))
    return squared_error / n_vals

  