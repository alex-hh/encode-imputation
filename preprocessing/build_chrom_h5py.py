import argparse
import h5py

from utils.CONSTANTS import all_chromosomes, n_expts, data_dir, BINNED_CHRSZ
from utils.track_handlers import TrackHandler
from utils.data_loaders import SeqMultiTargetGenerator


dataset = sys.argv[1]
chrom = sys.argv[2]

def main(dataset, chrom):
  # TODO - handle possibility of making dataset 'all' rather than train/val (see build joint chrom)
  # TODO - simplify track handler (or remove? it seems unnecessary for this purpose)
  chrom_len = BINNED_CHRSZ[chrom]
  n_obs = n_expts[dataset]

  print('Loading tracks', flush=True)
  th = TrackHandler(dataset=dataset, chroms=[chrom])
  # if dataset_dir is None:
  #     self.dataset_dir = dataset_dirs[dataset]
  #   else:
  #     self.dataset_dir = dataset_dir
  #   self.cell_types = cell_types
  #   self.assay_types = assay_types
  #   for expt_name in self.expt_names:
  #     cell_type = re.match('C\d{2}', expt_name).group(0) # overkill - could just slice
  #     assay_type = re.search('M\d{2}', expt_name).group(0)
  #     expt = Experiment(expt_name, self.dataset_dir,
  #                       chroms=chroms,
  #                       use_compressed=use_compressed,
  #                       transform=transform,
  #                       expt_suffix=expt_suffix)
  #     self.expts_by_type[cell_type].append(expt)
  #     self.expts_by_type[assay_type].append(expt)
  #     self.expts.append(expt)

  th.load_datasets()
  # self.data = defaultdict(dict)
  # for expt in self.expts:
  #   for chrom in self.chroms:
  #    self.data[expt.expt_name][chrom] = expt.load_chrnp(chrom)

  ## TODO - individual script for each expt?
  with h5py.File(data_dir + '{}_{}_targets.h5'.format(chrom, dataset), 'w') as h5f:
    h5f.create_dataset('targets', shape=(chrom_len, n_obs), chunks=(100, n_obs), compression='lzf')
    for i, expt_name in enumerate(th.expt_names):
      print(expt_name, flush=True)
      track = th.data[expt_name][chrom]
      print(track.nbytes, flush=True)
      h5f['targets'][:,i] = track


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('dataset', help='Dataset (train/val)')
  parser.add_argument('chrom', help='Chromosome string e.g. chr21') # a string
  args = parser.parse_args()
  main(args.dataset, args.chrom)