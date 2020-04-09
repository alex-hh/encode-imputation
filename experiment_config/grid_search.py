import copy
from utils.CONSTANTS import data_dir, BINNED_CHRSZ
from training.train_config_helpers import save_experiment_params


BASE_DROPFACT_SETTINGS = {'expt_name': 'test',
                          'model_kwargs': {'model_class': 'DAE',
                                           'obs_counts': [50, 45]},
                          'data_kwargs': {'data_class': 'HDF5InMemDict',
                                          'n_drop': 50,
                                          'n_predict': 50,
                                          'dataset': 'train',
                                          'directory': '/work/ahawkins/encodedata/',
                                          'chroms': ['chr21'],
                                          'chunks_per_batch': 256,
                                          'transform': None,
                                          'custom_splits': False,
                                          'dataset_fraction': 1, # number of samples to load into mem to constitute one 'epoch'
                                          'replace_gaps': False,
                                          'use_backup': False,
                                          'shuffle': True},
                          'val_kwargs': {'loss': 'mse',
                                         'shuffle': False,
                                         'eval_freq': 500000,
                                         'custom_val': True,
                                         'primary_metrics': ['low_supports', 'mid_supports', 'high_supports',
                                                             'high_mses', 'mid_mses'],
                                         'secondary_chroms': ['chr1', 'chr7', 'chr13', 'chr19'],
                                         'secondarychrom_val_size': 50000,
                                         'also_full_val': True,
                                         'dataset': 'val',
                                         'constant_size': False}, # these will override data kwargs
                          'train_kwargs': {'loss': 'mse',
                                           'optimizer': 'adam',
                                           'monitor_all': False,
                                           'lr': 3e-4,
                                           'n_samples': 30000000, # number of samples - here equivalent to 10 epochs 
                                           }}

base_region_params = copy.deepcopy(BASE_DROPFACT_SETTINGS)
base_region_params['model_kwargs']['model_class'] = 'MLPRegionDAE'
base_region_params['data_kwargs']['data_class'] = 'HDF5RegionInMem'
base_region_params['data_kwargs']['outputs_per_datapoint'] = 1
base_region_params['data_kwargs']['dataset_size'] = 500000
base_region_params['data_kwargs']['subsample_each_epoch'] = True
base_region_params['data_kwargs']['use_metaepochs'] = True
base_region_params['val_kwargs']['use_metaepochs'] = True

mlp_drops = [0.0, 0.3]
contexts = [None, 15]
n_drop = [25, 50, 100]
n_predict = [25, 50]
dae_dims = [25, [100,50]]
losses = ['mse', 'cauchy5']


# run_dropconfig = [(100, 25, 25), (50, 50, 25), (50, 25, 25), (100, 50, 25), (100, 25, [100, 50])]

# already_run = [(0.0, None, d, p, dd, 'cauchy5') for d,p,dd in zip(run_dropconfig)]
# already_run.append((0.0, None, 50, 50, 25, 'mse'))

point_pred_params = []
region_pred_params = []
for loss in losses:
  for mlpd in mlp_drops:
    for context in contexts:
      for n_dr in n_drop:
        for n_pred in n_predict:
          for dae_dim in dae_dims:
            params = [('tk', 'loss', loss), ('mk', 'mlp_dropout', mlpd), ('mk', 'dae_dim', dae_dim)]
            params += [('dk', 'n_drop', n_dr), ('mk', 'n_drop', n_dr), ('dk', 'n_predict', n_pred), ('mk', 'obs_counts', [n_pred, 45])]
            if context is not None:
              params += [('mk', 'bin_context_size', context), ('dk', 'context_size', context)]
              region_pred_params.append(params)
            else:
              point_pred_params.append(params)

save_experiment_params(BASE_DROPFACT_SETTINGS, point_pred_params, 'gridsearcholdsplits')
save_experiment_params(base_region_params, region_pred_params, 'regiongridsearcholdsplits')
base_region_params['val_kwargs']['secondary_chroms'] = []
base_region_params['val_kwargs']['custom_val'] = False
save_experiment_params(base_region_params, region_pred_params, 'regiongridsearcholdsplitsvalchr21only')

base_round1_winner_params = copy.deepcopy(BASE_DROPFACT_SETTINGS)
base_round1_winner_params['model_kwargs']['mlp_dropout'] = 0.3
base_round1_winner_params['model_kwargs']['dae_dim'] = [100,50]
base_round1_winner_params['data_kwargs']['n_drop'] = 50
base_round1_winner_params['data_kwargs']['n_predict'] = 50
base_round1_winner_params['model_kwargs']['n_drop'] = 50
base_round1_winner_params['model_kwargs']['obs_counts'] = [50, 45]
dae_dims = [100, 256, 512, [256,100], [256,256]]
dae_dim_params = []
for dae_dim in dae_dims:
  dae_dim_params.append([('mk', 'dae_dim', dae_dim), ('tk', 'loss', 'cauchy5')])
  dae_dim_params.append([('mk', 'dae_dim', dae_dim), ('tk', 'loss', 'mse')])
save_experiment_params(base_round1_winner_params, dae_dim_params, 'gridsearcholdsplitsdd')