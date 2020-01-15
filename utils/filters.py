import math
from collections import namedtuple, defaultdict

import pandas as pd
import numpy as np

from utils.track_calculations import get_top1_inds
from utils.CONSTANTS import all_chromosomes, CHRSZ, BINNED_CHRSZ, data_dir
from utils.interval_readers import load_bed, read_narrowpeak
from utils.track_handlers import Experiment

GenInterval = namedtuple('Interval', ['chrom', 'start', 'stop'])

# Question - how do we want the api to work for this
# one option is that a TrackHandler has a set of filters
# and these are automatically applied, so that the data loader
# just interfaces with the TrackHandler and not with the filters 
# directly. This would be quite nice. It just requires the TrackHandler
# to store filtered lengths for each chromosome.
# the problem we'd have though is that what we do with filtered regions
# might depend on parameters of the data loader
# i.e. if you want a context size of 1000, or of 10000 you might
# decide to include/exclude different windows based on the amount of 
# missing data
class IntervalFilter:
  """
   Store intervals of genomic coordinates which will be treated in a uniform way
    (e.g.) dropped from training / evaluation
   Blacklist, gaps, genome annotation could all be relevant.
  """
  def __init__(self, filename=None, chroms='all'):
    if chroms == 'all':
      chroms = all_chromosomes
    self.intervals = defaultdict(list)
    self.chroms = chroms
    self.filename = filename
    self.filtered_chrom_sizes = defaultdict(int)
    self.load()

  def __str__(self):
    return self.__class__.__name__

  def load(self):
    """
     populate self.intervals from self.filename
    """
    raise NotImplementedError()

  def filter_array(self, array, chrom, array_resolution=25):
    """
    Filter out an array of binned measurements
    """
    # TODO - use index arrays - https://stackoverflow.com/questions/28412538/pythonic-way-of-slicing-out-part-of-a-numpy-array

    retained_vals = []
    filter_end = 0
    for interval in self.intervals[chrom]:
      retained_vals += list(array[filter_end:interval.start//25])
      filter_end = interval.stop//25
    return np.asarray(retained_vals)

  def filter_bins(self, chrom, array_resolution=25):
    if self.filter_type == 'exclude':
      return self.exclude_bins(chrom, array_resolution=array_resolution)
    elif self.filter_type == 'select':
      return self.select_bins(chrom, array_resolution=array_resolution)

  def exclude_bins(self, chrom, array_resolution=25):
    retained_bins = []
    filter_end = 0
    for interval in self.intervals[chrom]:
      retained_bins += range(filter_end, interval.start//25)
      filter_end = math.ceil(interval.stop/25)
    return retained_bins

  def select_bins(self, chrom, array_resolution=25):
    bins = []
    for interval in self.intervals[chrom]:
      bins += range(interval.start//array_resolution, math.ceil(interval.stop/array_resolution)) # end is non inclusive - so bin after the final one in which the gene lies.
    return bins

  def filter_expt(self, expt):
    """
    Filter an Experiment
    """
    pass
    # use index arrays - https://stackoverflow.com/questions/28412538/pythonic-way-of-slicing-out-part-of-a-numpy-array

class MultiFilter(IntervalFilter):

  filter_type = 'select'

  def __init__(self, filters=[], chroms='all'):
    # select filters go first
    self.filters = sorted(filters, key=lambda x: x.filter_type, reverse=True)
    super().__init__(None)

  def select_bins(self, chrom, array_resolution=25, select_meth='union'):
    select_filters = [f for f in self.filters if f.filter_type == 'select']
    bins = set()
    if select_filters:
      for f in select_filters:
        if select_meth == 'union':
          bins |= set(f.select_bins(chrom))
        else:
          bins &= set(f.select_bins(chrom))
    else:
      bins = set(range(BINNED_CHRSZ[chrom]))
    exclude_filters = [f for f in self.filters if f.filter_type == 'exclude']
    for f in exclude_filters:
      to_drop = bins.intersection(set(f.select_bins(chrom)))
      bins -= to_drop
    return sorted(list(bins))

  def load(self):
    pass

class BinnedTrackMixin:

  def add_interval_by_bin_index(self, bin_ind):
    # take an index of a bin in a concatenated list of chromosome bins
    # and add the corresponding genomic interval
    for i, (chrom, cum_bins) in enumerate(zip(self.chroms, self.cum_bins_per_chrom)):
      if bin_ind < cum_bins:
        chrom_bin_index = bin_ind - self.cum_bins_per_chrom[i-1] if i > 0 else bin_ind
        self.intervals[chrom].append(GenInterval(chrom=chrom, start=chrom_bin_index*25, stop=(chrom_bin_index+1)*25))
        break
      if i == len(self.chroms) - 1:
        raise Error()

class RawVarFilter(BinnedTrackMixin, IntervalFilter):

  filter_type = 'select'

  def __init__(self, dataset='train', transform=None, chroms='all', frac=0.01, **kwargs):
    self.dataset = dataset
    self.transform = transform
    self.percentile = 100-(100*frac)
    if chroms == 'all':
      chroms = all_chromosomes
    bins_per_chrom = [BINNED_CHRSZ[chrom] for chrom in chroms]
    self.cum_bins_per_chrom = np.cumsum(bins_per_chrom)
    super().__init__(None, chroms=chroms, **kwargs)

  def load(self):
    print('Loading variance filter')
    all_var = []
    for chrom in self.chroms:
      chrom_var = np.load(data_dir+'{}_{}_var_{}.gz.npz'.format(chrom, self.dataset, self.transform))['arr_0']
      all_var.append(chrom_var)
    all_var = np.concatenate(all_var)
    top1inds = get_top1_inds(all_var, percentile=self.percentile)
    for bin_ind in top1inds:
      self.add_interval_by_bin_index(bin_ind)
    for chrom in self.chroms:
      self.filtered_chrom_sizes[chrom] = len(self.intervals[chrom])*25

      # todo find chromosome - c.f. cum_chunks_per_chrom
      # and convert from bin inds back to genome coordinates
      # double check that the intervals match the bin inds ultimately
      # i.e. filter_bins should return same as get_top1_inds on single chrom var track

class Top1Filter(IntervalFilter):

  filter_type = 'select'

  def __init__(self, expt_name, data_folder, frac=0.01, chroms='all', **kwargs):
    self.expt = Experiment(expt_name, data_folder, chroms=chroms)
    if chroms == 'all':
      chroms = all_chromosomes
    bins_per_chrom = [BINNED_CHRSZ[chrom] for chrom in chroms]
    self.cum_bins_per_chrom = np.cumsum(bins_per_chrom)
    self.percentile = 100-(100*frac)
    super().__init__(None, chroms=chroms, **kwargs)

  def load(self):
    print('Loading top 1 percent filter')
    expt = Experiment(self.expt_name)
    all_vals = expt.load_chroms()
    top1inds = get_top1_inds(all_vals, percentile=self.percentile)
    for bin_ind in top1inds:
      self.add_interval_by_bin_index(bin_ind)
    for chrom in self.chroms:
      self.filtered_chrom_sizes[chrom] = len(self.intervals[chrom])*25

class GapFilter(IntervalFilter):

  filter_type = 'exclude'
  
  def load(self):
    """
     Basically for the gap.txt files downloaded from goldenPath (hg - UCSC)
    """
    print('Loading gap filter')
    df = pd.read_csv('data/gap.txt', sep='\t',
                     names=['chromosome', 'start', 'end', 'chr_id',
                            'gap_char','gap_len', 'gap_type', '-'])
    # self.chrom_sizes[chrom] = CHRSZ[chrom]
    for i, chrom in enumerate(self.chroms):
      chrom_gaps = df[df['chromosome']==chrom]
      # print(len(chrom_gaps))
      for ix, row in chrom_gaps.iterrows():
        start, end = int(row['start']), int(row['end'])
        self.intervals[chrom].append(GenInterval(chrom=chrom, start=start, stop=end))
        self.filtered_chrom_sizes[chrom] -= end - start


class V2BlacklistFilter(IntervalFilter):

  filter_type = 'exclude'

  def load(self):
    print('Loading v2 blacklist filter')
    blacklist = load_bed('data/annotations/hg38.blacklist.v2.bed.gz')
    for line in blacklist:
      chrom, start, end, _ = line.split()
      start, end = int(start), int(end)
      self.intervals[chrom].append(GenInterval(chrom=chrom, start=start, stop=end))
      self.filtered_chrom_sizes[chrom] -= end-start

class BlacklistFilter(IntervalFilter):
  
  filter_type = 'exclude'

  def load(self):
    print('Loading blacklist filter')
    blacklist = load_bed('data/annotations/hg38.blacklist.bed.gz')
    for line in blacklist:
      chrom, start, end = line.split()
      start, end = int(start), int(end)
      self.intervals[chrom].append(GenInterval(chrom=chrom, start=start, stop=end))
      self.filtered_chrom_sizes[chrom] -= end-start


class GeneFilter(IntervalFilter):

  filter_type = 'select'

  def __init__(self, max_size=8000, **kwargs):
    self.max_size = max_size
    super().__init__(**kwargs)

  def load(self):
    print('Loading gene filter')
    gene_annot = load_bed('data/annotations/gencode.v29.genes.gtf.bed.gz')
    for line in gene_annot:
      chrom, start, end, _, _, strand = line.split()
      start, end = int(start), int(end)
      if end - start <= self.max_size:
        self.intervals[chrom].append(GenInterval(chrom=chrom, start=start, stop=end))
        self.filtered_chrom_sizes[chrom] += end - start

class EnhFilter(IntervalFilter):

  filter_type = 'select'

  def load(self):
    print('Loading enhancer filter')
    enh_annot = load_bed('data/annotations/F5.hg38.enhancers.bed.gz')
    for line in enh_annot:
      chrom, start, end, _, _, _, _, _, _, _, _, _ = line.split()
      start, end = int(start), int(end)
      self.intervals[chrom].append(GenInterval(chrom=chrom, start=start, stop=end))
      self.filtered_chrom_sizes[chrom] += end - start

class PromFilter(IntervalFilter):

  prom_loc = 2000
  filter_type = 'select'

  def load(self):
    print('Loading promoter filter')
    gene_annot = load_bed('data/annotations/gencode.v29.genes.gtf.bed.gz')
    for line in gene_annot:
      chrom, start, end, _, _, strand = line.split()
      start, end = int(start), int(end)
      if strand == '+':
        start = start-self.prom_loc
        end = start
      else:
        start = end
        end = end + self.prom_loc
      self.intervals[chrom].append(GenInterval(chrom=chrom, start=start, stop=end))
      self.filtered_chrom_sizes[chrom] += end - start