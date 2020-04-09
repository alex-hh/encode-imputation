import copy
from utils.CONSTANTS import data_dir, BINNED_CHRSZ, all_chromosomes
from training.expt_config_savers import save_experiment_params


BASE_DROPFACT_SETTINGS = {'expt_name': 'test',
                          'model_kwargs': {'model_class': 'DAE',
                                           'obs_counts': [20, 45],
                                           'n_drop': 100},
                          'data_kwargs': {'data_class': 'HDF5InMemDict',
                                          'n_drop': 100,
                                          'n_predict': 20,
                                          'dataset': 'train',
                                          'directory': '/work/ahawkins/encodedata/',
                                          'chroms': ['chr21'],
                                          'chunks_per_batch': 256,
                                          'transform': None,
                                          'custom_splits': False,
                                          'use_metaepochs': True,
                                          'subsample_each_epoch': True,
                                          # maybe look at avocado, avg, my models to see how consistent global rankings are across small samples
                                          'dataset_size': 1000000, # number of samples to load into mem to constitute one 'epoch'
                                          'replace_gaps': True,
                                          'use_backup': False,
                                          'shuffle': True},
                          'val_kwargs': {'data_class': 'HDF5InMemDict',
                                         'loss': 'mse', # what is this doing?
                                         'shuffle': False,
                                         'dataset': 'val',
                                         'train_dataset': 'all',
                                         'eval_freq': 500000,
                                         'custom_val': True,
                                         'trainchrom_val_size': 500000,
                                         'secondarychrom_val_size': 50000,
                                         'val_base_filename': '{}_selected_valbins.h5',
                                         'constant_size': False,
                                         'use_metaepochs': False,
                                         'subsample_each_epoch': False}, # these will override data kwargs
                          'train_kwargs': {'loss': 'mse',
                                           'optimizer': 'adam',
                                           'lr': 3e-4,
                                           'monitor_all': False,
                                           'n_samples': 30000000, # number of samples - here equivalent to 10 epochs 
                                           'same_seed': False}}


base_cauchy_settings = copy.deepcopy(BASE_DROPFACT_SETTINGS)
base_cauchy_settings['train_kwargs']['loss'] = 'cauchy5'
base_cauchy_settings['train_kwargs']['lr'] = 3e-4

base_cauchy_settings5050 = copy.deepcopy(base_cauchy_settings)
base_cauchy_settings5050['model_kwargs']['n_drop'] = 50
base_cauchy_settings5050['model_kwargs']['obs_counts'][0] = 50
base_cauchy_settings5050['data_kwargs']['n_predict'] = 50
base_cauchy_settings5050['data_kwargs']['n_drop'] = 50

base_mse_settings5050 = copy.deepcopy(base_cauchy_settings)
base_mse_settings5050['train_kwargs']['loss'] = 'mse'
base_mse_settings5050['model_kwargs']['n_drop'] = 50
base_mse_settings5050['model_kwargs']['obs_counts'][0] = 50
base_mse_settings5050['data_kwargs']['n_predict'] = 50
base_mse_settings5050['data_kwargs']['n_drop'] = 50

base_d100p20_settings = copy.deepcopy(base_cauchy_settings)
base_d100p20_settings['model_kwargs']['n_drop'] = 100
base_d100p20_settings['model_kwargs']['obs_counts'][0] = 20
base_d100p20_settings['data_kwargs']['n_predict'] = 20
base_d100p20_settings['data_kwargs']['n_drop'] = 100

base_new_splits_settings = copy.deepcopy(base_d100p20_settings)
base_new_splits_settings['data_kwargs']['data_class'] = 'HDF5InMemCustomSplits'
base_new_splits_settings['data_kwargs']['split_file'] = 'data/val_splits/splits_5fromM02_2toM02_2m2m.csv'
base_new_splits_settings['model_kwargs']['n_train_obs'] = 278
base_new_splits_settings['model_kwargs']['obs_counts'] = [20, 31]
del base_new_splits_settings['data_kwargs']['custom_splits']
# TODO - maybe just generate quickly a set of randomly sampled indexes for each chromosome
params = [[('dk', 'chroms', [c])] for c in all_chromosomes]
save_experiment_params(base_d100p20_settings, params, 'eachchrom')
save_experiment_params(base_new_splits_settings, params, 'eachchromnewsplits')
save_experiment_params(base_cauchy_settings5050, params, 'eachchrom5050')
save_experiment_params(base_mse_settings5050, params, 'eachchrommse5050')

base_gridsearch_win_settings = copy.deepcopy(base_cauchy_settings)
base_gridsearch_win_settings['model_kwargs']['mlp_dropout'] = 0.3
base_gridsearch_win_settings['model_kwargs']['dae_dim'] = [100,50]
base_gridsearch_win_settings['model_kwargs']['obs_counts'] = [50, 45]
base_gridsearch_win_settings['model_kwargs']['n_drop'] = 50
base_gridsearch_win_settings['data_kwargs']['n_drop'] = 50
base_gridsearch_win_settings['data_kwargs']['n_predict'] = 50
save_experiment_params(base_gridsearch_win_settings, params, 'eachchromgridwinner')

test_gw_settings = copy.deepcopy(base_gridsearch_win_settings)
test_gw_settings['data_kwargs']['dataset'] = 'all'
test_gw_settings['model_kwargs']['n_train_obs'] = 312
test_gw_settings['val_kwargs']['no_val'] = True
del test_gw_settings['val_kwargs']['trainchrom_val_size']
del test_gw_settings['val_kwargs']['secondarychrom_val_size']
del test_gw_settings['val_kwargs']['val_base_filename']
save_experiment_params(test_gw_settings, params, 'eachchromgridwinnertest')

gridwinewatest_settings = copy.deepcopy(test_gw_settings)
gridwinewatest_settings['val_kwargs']['weighted_average'] = True
save_experiment_params(gridwinewatest_settings, params, 'eachchromgridwinnerewatest')

ewa_secondary_chroms = ['chr1', 'chr7', 'chr13', 'chrX']
gridwin_ewa_settings = copy.deepcopy(base_gridsearch_win_settings)
gridwin_ewa_settings['val_kwargs']['weighted_average'] = True
params = [[('dk', 'chroms', [c]), ('val_kwargs', 'secondary_chroms', ewa_secondary_chroms+[c] if c not in ewa_secondary_chroms else ewa_secondary_chroms)] for c in all_chromosomes]
save_experiment_params(gridwin_ewa_settings, params, 'eachchromgridwinnerewa')


base_test_settings = copy.deepcopy(base_new_splits_settings)
base_test_settings['data_kwargs']['dataset'] = 'all'
base_test_settings['model_kwargs']['n_train_obs'] = 312
base_test_settings['val_kwargs']['no_val'] = True

del base_test_settings['val_kwargs']['trainchrom_val_size']
del base_test_settings['val_kwargs']['secondarychrom_val_size']
del base_test_settings['val_kwargs']['val_base_filename']


save_experiment_params(base_test_settings, params, 'eachchromfulltrain')
