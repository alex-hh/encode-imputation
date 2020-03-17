import copy
from training.expt_config_savers import save_train_config
from experiment_config.base_settings import COMMON_BASE_SETTINGS


# DATA KWARGS SUPP
OLD_DATA_SUPP = {'data_class': 'HDF5InMemDict',
                 'n_predict': 50,
                 'chroms': ['chr21'],
                 'chunks_per_batch': 256,
                 'transform': None,
                 'custom_splits': False,
                 'use_metaepochs': True,
                 'subsample_each_epoch': True,
                  # maybe look at avocado, avg, my models to see how consistent global rankings are across small samples
                 'dataset_size': 1000000, # number of samples to load into mem to constitute one 'epoch'
                 'replace_gaps': True,
                 'use_backup': False}

# note lack of gap replacement in the fully inmem data generator
NEW_INMEM_SUPP = {'data_class': 'TrainDataGeneratorHDF5',
                  'batch_size': 256,
                  'split_file': None,
                  'replace_gaps': True,
                  'chrom': 'chr21'}

NEW_CHUNKED_SUPP = {'data_class': 'ChunkedTrainDataGeneratorHDF5',
                    'dataset_size': 1000000,
                    'replace_gaps': True,
                    'split_file': None,
                    'chrom': 'chr21'}

# TODO - handle directory as env var? or via settings?
gdrive_dir = '/content/drive/My Drive/enc_datacore/encodeimp/data/'
cluster_dir = '/work/ahawkins/encodedata/'

new_chunked_sett = copy.deepcopy(COMMON_BASE_SETTINGS)
new_chunked_sett['data_kwargs'].update(NEW_CHUNKED_SUPP)
save_train_config('chunkedtrain', new_chunked_sett, 'chr21_reprod')

new_inmem_sett = copy.deepcopy(COMMON_BASE_SETTINGS)
new_inmem_sett['data_kwargs'].update(NEW_INMEM_SUPP)
save_train_config('inmemtrain', new_inmem_sett, 'chr21_reprod')

old_sett = copy.deepcopy(COMMON_BASE_SETTINGS)
old_sett['data_kwargs'].update(OLD_DATA_SUPP)
old_sett['train_kwargs']['n_samples'] = 14000000
# save_train_config('chunkedtrain_cluster', old_sett, 'chr21_reprod')
# old_sett['data_kwargs']['directory'] = gdrive_dir
save_train_config('hdf5inmem', old_sett, 'chr21_reprod')