import gzip
import numpy as np
from collections import namedtuple, defaultdict

peak_cols = ['chrom', 'start', 'stop', 'p', 'summit']
Peak = namedtuple('Peak', peak_cols)

Interval = namedtuple('Interval', ['chrom', 'start', 'stop',
                                   'left_pad', 'right_pad'])

## Peaks aren't exactly the same as what encode reports but this is a reasonable start...it's prob just a question of the params (-g, -l)

def read_narrowpeak(filename):
  interval_dict = defaultdict(list)
  with open(filename, 'r') as bdg:
    for line in bdg:

      vals = line.split('\t')
      # chr4  6782873 6783604 Peak_1  1000  . 30.09811  104.01200 94.52104  367
      colnames = ['chrom', 'start', 'stop', 'name', 'strength', 'na', 'fold_change', 'p', 'q', 'summit']
      # for c, v in zip(colnames, vals):
        # if c in peak_cols:
      peak = Peak(**{c: v for c, v in zip(colnames, vals) if c in peak_cols})
      peak.start = int(peak.start)
      peak.stop = int(peak.stop)
      try:
        peak.p = float(peak.p)
      except:
        peak.p = np.nan
      interval_dict[peak.chrom].append(peak)
  return interval_dict


def load_bed(bed):
  """Read gzipped/uncompressed BED
  """
  # log.info('Reading from BED {}...'.format(bed))
  result = []
  if bed.endswith('gz'):
    with gzip.open(bed, 'r') as infile:
      for line in infile:
        result.append(line.decode("ascii"))
  else:
    with open(bed, 'r') as infile:
      for line in infile:
        result.append(line)
  return result