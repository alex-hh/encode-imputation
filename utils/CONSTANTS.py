import os, re
import pandas as pd
# chromosome lengths: cat hg38.chrom.sizes | grep -P "chr[\dX]" | grep -v _
# c21 5010000

m2name = {'M01': 'ATAC-seq',
          'M02': 'DNase-seq',
          'M03': 'H2AFZ',
          'M04': 'H2AK5ac',
          'M05': 'H2AK9ac',
          'M06': 'H2BK120ac',
          'M07': 'H2BK12ac',
          'M08': 'H2BK15ac',
          'M09': 'H2BK20ac',
          'M10': 'H2BK5ac',
          'M11': 'H3F3A',
          'M12': 'H3K14ac',
          'M13': 'H3K18ac',
          'M14': 'H3K23ac',
          'M15': 'H3K23me2',
          'M16': 'H3K27ac',
          'M17': 'H3K27me3',
          'M18': 'H3K36me3',
          'M19': 'H3K4ac',
          'M20': 'H3K4me1',
          'M21': 'H3K4me2',
          'M22': 'H3K4me3',
          'M23': 'H3K56ac',
          'M24': 'H3K79me1',
          'M25': 'H3K79me2',
          'M26': 'H3K9ac',
          'M27': 'H3K9me1',
          'M28': 'H3K9me2',
          'M29': 'H3K9me3',
          'M30': 'H3T11ph',
          'M31': 'H4K12ac',
          'M32': 'H4K20me1',
          'M33': 'H4K5ac',
          'M34': 'H4K8ac',
          'M35': 'H4K91ac'}

CHRSZ = {
    'chr1': 248956422,
    'chr10': 133797422,
    'chr11': 135086622,
    'chr12': 133275309,
    'chr13': 114364328,
    'chr14': 107043718,
    'chr15': 101991189,
    'chr16': 90338345,
    'chr17': 83257441,
    'chr18': 80373285,
    'chr19': 58617616,
    'chr2': 242193529,
    'chr20': 64444167,
    'chr21': 46709983,
    'chr22': 50818468,
    'chr3': 198295559,
    'chr4': 190214555,
    'chr5': 181538259,
    'chr6': 170805979,
    'chr7': 159345973,
    'chr8': 145138636,
    'chr9': 138394717,
    'chrX': 156040895
}

BINNED_CHRSZ = {'chr1': 9958257,
                'chr10': 5351897,
                'chr11': 5403465,
                'chr12': 5331013,
                'chr13': 4574574,
                'chr14': 4281749,
                'chr15': 4079648,
                'chr16': 3613534,
                'chr17': 3330298,
                'chr18': 3214932,
                'chr19': 2344705,
                'chr2': 9687742,
                'chr20': 2577767,
                # 'chr21': 1283,
                # 'chr21': 1000000,
                'chr21': 1868400,
                'chr22': 2032739,
                'chr3': 7931823,
                'chr4': 7608583,
                'chr5': 7261531,
                'chr6': 6832240,
                'chr7': 6373839,
                'chr8': 5805546,
                'chr9': 5535789,
                'chrX': 6241636}

all_chromosomes = ['chr{}'.format(i) for i in list(range(1,23))] + ['chrX']

data_dir = os.environ.get('DATA_DIR', 'data/')
config_dir = os.environ.get('CONFIG_DIR', 'experiment_config/')
output_dir = os.environ.get('OUTPUT_DIR', 'outputs/')
# if data_dir in ['data', 'data/']:
#     genome_path = 'remote_data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta'
# else:
#     genome_path = '/work/ahawkins/encodedata/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta'

n_train_expts = 267
n_val_expts = 45
n_expts = {'train': 267, 'val': 45}

train_df = pd.read_csv(os.path.join(data_dir, 'enhanced_metadata_training_data.tsv'))
train_df.set_index('filename', inplace=True)
train_syn_ids = train_df['synapse_id']

all_metadata = pd.read_csv(os.path.join(data_dir, 'metadata_all.csv'))
test_expt_names = all_metadata[all_metadata['set']=='B']['track_code'].values

train_expt_names = [f.split('.')[0] for f in train_df.index.values]
val_expt_names = [f.split('.')[0] for f in pd.read_csv(os.path.join(data_dir,'metadata_validation_data.tsv'), sep='\t')['filename'].values]
dataset_expts = {'train': train_expt_names,
                 'val': val_expt_names,
                 'all': train_expt_names + val_expt_names,
                 'test': test_expt_names}

def cell_assay_types_from_expts(expt_names):
  cell_types = sorted(list(set([re.match('C\d{2}', e).group(0) for e in expt_names])))
  assay_types = sorted(list(set([re.search('M\d{2}', e).group(0) for e in expt_names])))
  return cell_types, assay_types

dataset_celltypes = {}
dataset_assaytypes = {}
for k, expts in dataset_expts.items():
    cell_types, assay_types = cell_assay_types_from_expts(expts)
    dataset_celltypes[k] = cell_types
    dataset_assaytypes[k] = assay_types

train_cell2id = {c: i for i, c in enumerate(dataset_celltypes['train'])}
train_assay2id = {a: i for i, a in enumerate(dataset_assaytypes['train'])}

LOCAL = data_dir in ['data', 'data/']
